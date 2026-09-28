#!/usr/bin/env python3
"""Reproduce corrected IHDP and Minari HalfCheetah SPRL benchmarks.

Run from the release root, for example:
  python -m experiments.benchmarks --benchmark ihdp --ihdp-data /path/to/ihdp_npci_1.csv
  python -m experiments.benchmarks --benchmark halfcheetah

The default protocol is the four-method neural-teacher confirmation at seeds
101--200. External data are loaded only for the requested benchmark. --smoke
uses seed 101 and short non-paper training by default; it is only an
installation check.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
import torch
from sklearn.linear_model import LogisticRegression

from dgp.benchmark_support import (EmpiricalShiftSupport, analytic_ihdp_target,
                                   load_ihdp_covariates, sample_ihdp_with_subjects)
from dgp.ihdp_bandit import IHDPConfoundedBandit
from estimators.robust_ipw_continuous import robust_ipw_bounds_samplewise
from experiments.benchmark_utils import (additive_predictions, call_recorded,
                                         fit_additive_readout, fit_interaction_readout,
                                         predict_model, probability_metrics,
                                         readout_predictions, split_halfcheetah)
from representations.balanced import BalancedRep, train_balanced
from representations.propensity import PropensityModel, get_propensities, train_propensity
from representations.sprl_cell import SPRL, train_sprl

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/benchmarks.json").read_text(encoding="utf-8"))
METHODS = tuple(CONFIG["methods"])


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def atomic_json(path: Path, value) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".tmp", delete=False, encoding="utf-8") as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def case_dir(root: Path, benchmark: str, w_u: float, seed: int) -> Path:
    return root / benchmark / f"wu_{round(w_u * 10):02d}" / f"seed_{seed:03d}"


def complete(path: Path) -> bool:
    if not (path / "result.json").is_file():
        return False
    needed = [path / f"{label}_observations.npz" for label in ("train", "fresh")]
    needed += [path / f"{method}_fresh_predictions.npz" for method in METHODS]
    needed += [path / f"{method}.pt" for method in METHODS if method != "raw"]
    return all(p.is_file() for p in needed)


def software_profile() -> dict:
    return {"platform": platform.platform(), "python": sys.version,
            "torch": torch.__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
            "device": "cpu", "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads()}


def configure_torch() -> None:
    training = CONFIG["training"]
    threads = int(training["torch_threads"])
    interop = int(training["torch_interop_threads"])
    if torch.get_num_threads() != threads:
        torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != interop:
        torch.set_num_interop_threads(interop)


def make_ihdp_case(x: np.ndarray, support: EmpiricalShiftSupport,
                   w_u: float, seed: int, output: Path, *, smoke: bool = False) -> dict:
    dgp = IHDPConfoundedBandit(x, n_instrument=10, w_u=w_u,
                               conf_strength=2.0, seed=seed)
    train = sample_ihdp_with_subjects(dgp, 128 if smoke else 5000, support)
    fresh_dgp = IHDPConfoundedBandit(x, n_instrument=10, w_u=w_u,
                                     conf_strength=2.0, seed=seed)
    fresh_dgp.rng = np.random.RandomState(20260925 + 300000 + 1000 * round(10 * w_u) + seed)
    fresh = sample_ihdp_with_subjects(fresh_dgp, 128 if smoke else 10000, support)
    for label, data in (("train", train), ("fresh", fresh)):
        np.savez_compressed(output / f"{label}_observations.npz", **data)
    target = analytic_ihdp_target(dgp)
    if not np.isclose(target, 0.3, rtol=0, atol=1e-12):
        raise ValueError(f"IHDP analytic target differs from 0.3: {target}")
    return {"train": train, "fit": train, "fresh": fresh, "target": target,
            "gamma_reference": float(np.exp(abs(w_u))),
            "support_overlap": support.overlap_points, "dgp": dgp}


def save_model(model: torch.nn.Module, path: Path) -> None:
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, path)


def fit_evaluation_propensity(z: np.ndarray, a: np.ndarray, path: Path):
    model = LogisticRegression(C=1.0, max_iter=1000)
    _, seen = call_recorded(model.fit, z, a)
    joblib.dump(model, path)
    return model, seen


def train_case(benchmark: str, w_u: float, seed: int, output: Path,
               *, x: np.ndarray | None = None,
               support: EmpiricalShiftSupport | None = None,
               episodes: list[dict] | None = None, smoke: bool = False) -> dict:
    """One paired seed: common observations, four fixed method configurations."""
    began = time.monotonic()
    if benchmark == "ihdp":
        if x is None or support is None:
            raise ValueError("IHDP covariates and empirical support are required")
        case = make_ihdp_case(x, support, w_u, seed, output, smoke=smoke)
    elif benchmark == "halfcheetah":
        if episodes is None or w_u != 0.5:
            raise ValueError("HalfCheetah requires episodes and w_u=0.5")
        case = split_halfcheetah(episodes, seed, output, smoke=smoke)
    else:
        raise ValueError(f"Unknown benchmark {benchmark}")
    train, fit, fresh = case["train"], case["fit"], case["fresh"]
    s, a, r = (train[k] for k in ("s", "a", "r"))
    sf, af, rf, uf = (fresh[k] for k in ("s", "a", "r", "u"))
    dim = s.shape[1]
    latent_dim = 2 if benchmark == "ihdp" else 4
    transitions = benchmark == "halfcheetah"
    sn = train["s_next"] if transitions else None
    epochs = 1 if smoke else 100
    warnings_all, histories = {}, {}

    torch.manual_seed(seed + 9000)
    teacher = PropensityModel(dim, hidden_dim=256)
    save_model(teacher, output / "teacher_neural_initial.pt")
    histories["teacher_neural"], warnings_all["teacher_neural_training"] = call_recorded(
        train_propensity, teacher, s, a, epochs=epochs, device="cpu", verbose=False)
    save_model(teacher, output / "teacher_neural.pt")
    write_json(output / "teacher_neural_history.json", histories["teacher_neural"])
    e_teacher = get_propensities(teacher, s, "cpu")
    e_teacher_fresh = get_propensities(teacher, sf, "cpu")
    np.save(output / "teacher_neural_train.npy", e_teacher)
    np.save(output / "teacher_neural_fresh.npy", e_teacher_fresh)

    torch.manual_seed(seed + 4000)
    balanced = BalancedRep(dim, latent_dim).to("cpu")
    save_model(balanced, output / "balanced_initial.pt")
    histories["balanced"], warnings_all["balanced_training"] = call_recorded(
        train_balanced, balanced, s, a, r, alpha_mmd=0.1,
        epochs=epochs, device="cpu", verbose=False)
    save_model(balanced, output / "balanced.pt")
    write_json(output / "balanced_history.json", histories["balanced"])

    torch.manual_seed(seed + 5000)
    world = SPRL(dim, latent_dim, hidden_dim=256, use_transitions=transitions).to("cpu")
    save_model(world, output / "world_model_initial.pt")
    histories["world_model"], warnings_all["world_model_training"] = call_recorded(
        train_sprl, world, s, a, r, np.zeros(len(s)), data_s_next=sn,
        lambda_prop=0.0, epochs=epochs, warmup_epochs=0, lambda_ramp_epochs=0,
        device="cpu", verbose=False)
    save_model(world, output / "world_model.pt")
    write_json(output / "world_model_history.json", histories["world_model"])

    torch.manual_seed(seed + 6000)
    sprl = SPRL(dim, latent_dim, hidden_dim=256, use_transitions=transitions).to("cpu")
    save_model(sprl, output / "sprl_neural_initial.pt")
    histories["sprl_neural"], warnings_all["sprl_neural_training"] = call_recorded(
        train_sprl, sprl, s, a, r, e_teacher, data_s_next=sn,
        lambda_prop=1.0, sigma=None, epochs=epochs,
        warmup_epochs=0 if smoke else 25, lambda_ramp_epochs=1 if smoke else 25,
        grad_clip=1.0, device="cpu", verbose=False)
    save_model(sprl, output / "sprl_neural.pt")
    write_json(output / "sprl_neural_history.json", histories["sprl_neural"])

    models = {"balanced": balanced, "world_model": world, "sprl_neural": sprl}
    results = {}
    for name in METHODS:
        model = models.get(name)
        if model is None:
            train_pred, fit_pred, fresh_pred = ({"z": s}, {"z": fit["s"]}, {"z": sf})
        else:
            train_pred = predict_model(model, s, a, sn)
            fit_pred = (train_pred if benchmark == "ihdp" else
                        predict_model(model, fit["s"], fit["a"], fit["s_next"]))
            fresh_pred = predict_model(model, sf, af,
                                       fresh["s_next"] if transitions else None)
        zt, zfit, zf = train_pred["z"], fit_pred["z"], fresh_pred["z"]
        np.save(output / f"{name}_train_z.npy", zt)
        np.save(output / f"{name}_fit_z.npy", zfit)
        if benchmark == "ihdp":
            scaler, readout, seen = fit_additive_readout(
                zfit, fit["a"], fit["u"], output / f"{name}_readout.joblib")
            fit_readout = additive_predictions(scaler, readout, zfit, fit["u"])
            fresh_readout = additive_predictions(scaler, readout, zf, uf)
        else:
            scaler, readout, seen = fit_interaction_readout(
                zfit, fit["a"], fit["u"], output / f"{name}_readout.joblib")
            fit_readout = readout_predictions(scaler, readout, zfit, fit["u"])
            fresh_readout = readout_predictions(scaler, readout, zf, uf)
        warnings_all[f"{name}_readout_fit"] = seen
        saved = {**fresh_pred, "p_action_readout": fresh_readout["p_actual"],
                 "odds_ratio": fresh_readout["odds_ratio"]}
        if transitions:
            saved.update({key: fresh_readout[key] for key in
                          ("p_u0_raw", "p_u1_raw", "p_u0_clipped", "p_u1_clipped")})
        metrics = {
            "fit_amplification_median": fit_readout["gamma_z_median"] / case["gamma_reference"],
            "fresh_amplification_median": fresh_readout["gamma_z_median"] / case["gamma_reference"],
            "fit_amplification_p95": fit_readout["gamma_z_p95"] / case["gamma_reference"],
            "fresh_amplification_p95": fresh_readout["gamma_z_p95"] / case["gamma_reference"],
            "fit_counterfactual_probability_clipped": fit_readout["n_counterfactual_probability_clipped"],
            "fresh_counterfactual_probability_clipped": fresh_readout["n_counterfactual_probability_clipped"],
            "fresh_readout_action": probability_metrics(af, fresh_readout["p_actual"]),
            "fresh_reward_mse": None, "fresh_transition_mse": None,
            "fresh_latent_target_second_moment": None,
            "fresh_latent_target_variance": None,
        }
        if model is not None:
            metrics["fresh_reward_mse"] = float(np.mean((fresh_pred["reward_prediction"] - rf) ** 2))
            if benchmark == "ihdp":
                with torch.no_grad():
                    target_action_one = model.reward_predictor(
                        torch.as_tensor(zf, dtype=torch.float32),
                        torch.ones(len(zf), dtype=torch.float32)).numpy()
                true_action_one = sf @ case["dgp"].w_r + case["dgp"].w_ra
                metrics["fresh_action_one_mean_mse"] = float(np.mean((target_action_one - true_action_one) ** 2))
                saved["reward_action_one_prediction"] = target_action_one
            if "latent_next_target" in fresh_pred:
                target_latent = fresh_pred["latent_next_target"]
                metrics["fresh_transition_mse"] = float(np.mean((fresh_pred["latent_next_prediction"] - target_latent) ** 2))
                metrics["fresh_latent_target_second_moment"] = float(np.mean(target_latent ** 2))
                metrics["fresh_latent_target_variance"] = float(np.mean(np.var(target_latent, axis=0)))
        if benchmark == "ihdp":
            if name == "raw":
                ef, eval_warnings = fresh["pi_b_observed"], []
            else:
                propensity, eval_warnings = fit_evaluation_propensity(
                    zt, a, output / f"{name}_evaluation_propensity.joblib")
                ef = propensity.predict_proba(zf)[:, 1]
            warnings_all[f"{name}_evaluation_propensity_fit"] = eval_warnings
            saved["evaluation_e1"] = ef
            pi_e_a = (af == 1).astype(float)
            lower, upper, _ = robust_ipw_bounds_samplewise(
                af, rf, pi_e_a, case["gamma_reference"], ef)
            metrics.update({"interval_lower": float(lower), "interval_upper": float(upper),
                            "interval_width": float(upper - lower),
                            "includes_analytic_target": bool(lower <= case["target"] <= upper),
                            "evaluation_propensity_clipped": int(np.count_nonzero(
                                (ef < 1e-8) | (ef > 1 - 1e-8)))})
        np.savez_compressed(output / f"{name}_fresh_predictions.npz", **saved)
        results[name] = metrics

    result = {"benchmark": benchmark, "w_u": w_u, "seed": seed,
              "smoke_nonpaper": smoke, "n_training": int(len(s)),
              "n_readout_fit": int(len(fit["s"])), "n_fresh": int(len(sf)),
              "gamma_reference": case["gamma_reference"],
              "analytic_target": case["target"],
              "teacher_neural_fresh_action": probability_metrics(af, e_teacher_fresh),
              "methods": results, "warnings": warnings_all,
              "elapsed_seconds": time.monotonic() - began}
    if benchmark == "ihdp":
        result["support_overlap_points"] = case["support_overlap"]
        result["fresh_teacher_oracle_rmse"] = {"neural": float(np.sqrt(np.mean(
            (e_teacher_fresh - fresh["pi_b_observed"]) ** 2)))}
    else:
        result["episode_split"] = case["episode_split"]
    write_json(output / "result.json", result)
    return result


def progress(root: Path, benchmark: str) -> dict:
    first_seed, last_seed = CONFIG["seeds"]
    counts = {str(w): sum(complete(case_dir(root, benchmark, w, seed))
                          for seed in range(first_seed, last_seed + 1))
              for w in CONFIG["conditions"][benchmark]}
    value = {"updated_utc": stamp(), "benchmark": benchmark,
             "complete_cases": counts, "total_complete": sum(counts.values()),
             "target": len(counts) * (last_seed - first_seed + 1),
             "first_seed": first_seed, "last_seed": last_seed}
    atomic_json(root / "progress.json", value)
    return value


def resolve_seed_range(first_seed: int, last_seed: int | None,
                       smoke: bool) -> tuple[int, int]:
    if last_seed is None:
        last_seed = first_seed if smoke else 200
    if not 101 <= first_seed <= last_seed <= 200:
        raise ValueError("The confirmed protocol uses seed IDs 101 through 200")
    return first_seed, last_seed


def prepare_output_dir(root: Path) -> None:
    """Avoid adopting an unrelated output directory as a benchmark run."""
    if root.exists():
        if not root.is_dir():
            raise NotADirectoryError(root)
        if any(root.iterdir()) and not (root / "run_configuration.json").is_file():
            raise ValueError(f"Nonempty output has no benchmark configuration: {root}")
    else:
        root.mkdir(parents=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=("ihdp", "halfcheetah"))
    parser.add_argument("--first-seed", type=int, default=101)
    parser.add_argument("--last-seed", type=int,
                        help="Last seed (default: 200, or first seed with --smoke)")
    parser.add_argument("--output", type=Path, help="Output directory (default: outputs/benchmarks/<benchmark>)")
    parser.add_argument("--ihdp-data", type=Path, help="Local CEVAE ihdp_npci_1.csv file")
    parser.add_argument("--smoke", action="store_true",
                        help="Short non-paper execution check; one seed by default")
    args = parser.parse_args()
    try:
        args.first_seed, args.last_seed = resolve_seed_range(
            args.first_seed, args.last_seed, args.smoke)
    except ValueError as exc:
        parser.error(str(exc))
    if args.benchmark == "ihdp" and args.ihdp_data is None:
        parser.error("--ihdp-data is required for IHDP; data are not distributed")
    root = (args.output or ROOT / "outputs/benchmarks" / args.benchmark).resolve()
    prepare_output_dir(root)
    run_config = {"benchmark": args.benchmark, "seeds": CONFIG["seeds"],
                  "smoke_nonpaper": args.smoke,
                  "protocol": CONFIG, "ihdp_data": str(args.ihdp_data.resolve())
                  if args.ihdp_data else None}
    config_path = root / "run_configuration.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != run_config:
            raise ValueError("Output has a different run configuration; choose another --output")
    else:
        atomic_json(config_path, run_config)
    configure_torch()
    profile_path = root / "software_device.json"
    if not profile_path.exists():
        atomic_json(profile_path, {**software_profile(), "created_utc": stamp()})
    with (root / ".run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Benchmark already running for this output") from exc
        x = load_ihdp_covariates(args.ihdp_data) if args.benchmark == "ihdp" else None
        support = EmpiricalShiftSupport(x, 10, 2.0) if x is not None else None
        if args.benchmark == "halfcheetah":
            from dgp.confounded_halfcheetah import load_halfcheetah_episodes
            episodes = load_halfcheetah_episodes(max_episodes=7 if args.smoke else 1000)
        else:
            episodes = None
        for w_u in CONFIG["conditions"][args.benchmark]:
            for seed in range(args.first_seed, args.last_seed + 1):
                final = case_dir(root, args.benchmark, w_u, seed)
                if complete(final):
                    continue
                if final.exists():
                    raise ValueError(f"Incomplete case requires inspection: {final}")
                final.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix=f".seed_{seed:03d}_attempt_",
                                                 dir=final.parent) as tmp:
                    try:
                        print(f"Starting {args.benchmark} w_u={w_u} seed={seed}", flush=True)
                        result = train_case(args.benchmark, w_u, seed, Path(tmp),
                                            x=x, support=support, episodes=episodes,
                                            smoke=args.smoke)
                        Path(tmp).rename(final)
                    except Exception as exc:
                        with (root / "technical_failures.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps({"utc": stamp(), "benchmark": args.benchmark,
                                                     "w_u": w_u, "seed": seed,
                                                     "error": repr(exc),
                                                     "traceback": traceback.format_exc()}) + "\n")
                        raise
                state = progress(root, args.benchmark)
                print(f"Completed seed {seed}; elapsed={result['elapsed_seconds']:.1f}s; "
                      f"progress={state['total_complete']}/{state['target']}", flush=True)
        progress(root, args.benchmark)


if __name__ == "__main__":
    main()
