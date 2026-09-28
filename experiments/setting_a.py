"""Setting A: one-step empirical target inclusion (100 seeds per strength).

Run from the release root: python -m experiments.setting_a --output outputs/setting_a
``--smoke`` is a fast non-paper execution check. All runs use CPU to match
the verified learned-method replay. Existing seed files are never overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dgp.unified_cmdp import UnifiedCMDP
from estimators.robust_ipw_continuous import estimate_propensity, robust_ipw_bounds_samplewise
from representations.ae import AE, get_ae_encoder_fn, train_ae
from representations.balanced import BalancedRep, get_balanced_encoder_fn, train_balanced
from representations.contrastive import ContrastiveModel, get_contrastive_encoder_fn, train_contrastive
from representations.propensity import PropensityModel, get_propensities, train_propensity
from representations.sprl_cell import SPRL, get_sprl_encoder_fn, train_sprl
from representations.vae import VAE, get_vae_encoder_fn, train_vae
from utils.runtime import configure_cpu

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "setting_a.json"


def load_config(smoke: bool = False) -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if smoke:
        config = {**config, "run_type": "SMOKE / NON-PAPER OVERRIDE",
                  "w_u": [0.5], "n_samples": 128, "epochs": 1,
                  "propensity_epochs": 1, "warmup_epochs": 0,
                  "lambda_ramp_epochs": 1}
    else:
        config["run_type"] = "paper protocol"
    return config


def run_seed(config: dict, w_u: float, seed: int) -> dict:
    dgp = UnifiedCMDP(w_u=w_u, **config["dgp"])
    target = float(config["analytic_target_value"])
    if not np.isclose(dgp.one_step_always_one_value(), target, atol=1e-12, rtol=0):
        raise ValueError("Setting A analytic target does not match the DGP")
    gamma = float(np.exp(abs(w_u)))
    if not np.isclose(dgp.gamma_s, gamma, atol=1e-12, rtol=0):
        raise ValueError("Setting A sensitivity does not match the DGP")

    # Same sampling and teacher order as the verified replay.
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    data = dgp.sample_one_step(config["n_samples"], rng=rng)
    s, a, r = (data[key] for key in ("s", "a", "r"))
    pi_e_a = (a == 1).astype(float)
    offsets = config["initialization_seed_offsets"]
    torch.manual_seed(seed + offsets["teacher"])
    teacher = PropensityModel(s.shape[1]).to("cpu")
    train_propensity(teacher, s, a, epochs=config["propensity_epochs"],
                     device="cpu", verbose=False)
    teacher_e = get_propensities(teacher, s, device="cpu")

    def evaluate(method: str, e: np.ndarray) -> dict:
        low, high, _ = robust_ipw_bounds_samplewise(a, r, pi_e_a, gamma, e)
        if not np.isfinite(low) or not np.isfinite(high) or low > high:
            raise ValueError(f"Invalid interval for {method}, w_u={w_u}, seed={seed}")
        return {"lower": low, "upper": high, "width": high - low,
                "includes_target": bool(low <= target <= high)}

    rows = {"raw": evaluate("raw", dgp.pi_b_marg(s))}
    d, dz, epochs = s.shape[1], config["latent_dim"], config["epochs"]
    specs = (
        ("ae", seed + offsets["ae"], lambda: AE(d, dz),
         lambda m: train_ae(m, s, epochs=epochs, device="cpu", verbose=False), get_ae_encoder_fn),
        ("vae", seed + offsets["vae"], lambda: VAE(d, dz),
         lambda m: train_vae(m, s, epochs=epochs, device="cpu", verbose=False), get_vae_encoder_fn),
        ("contrastive", seed + offsets["contrastive"], lambda: ContrastiveModel(d, dz),
         lambda m: train_contrastive(m, s, epochs=epochs,
                                     noise_std=config["contrastive_noise_std"],
                                     device="cpu", verbose=False), get_contrastive_encoder_fn),
        ("balanced", seed + offsets["balanced"], lambda: BalancedRep(d, dz),
         lambda m: train_balanced(m, s, a, r, alpha_mmd=config["balanced_alpha_mmd"],
                                  epochs=epochs, device="cpu", verbose=False), get_balanced_encoder_fn),
        ("world_model", seed + offsets["world_model"], lambda: SPRL(d, dz, use_transitions=False),
         lambda m: train_sprl(m, s, a, r, e_hat=np.zeros(len(s)), lambda_prop=0.0,
                              epochs=epochs, warmup_epochs=0, lambda_ramp_epochs=0,
                              device="cpu", verbose=False), get_sprl_encoder_fn),
        ("sprl", seed + offsets["sprl"], lambda: SPRL(d, dz, use_transitions=False),
         lambda m: train_sprl(m, s, a, r, e_hat=teacher_e,
                              lambda_prop=config["lambda_prop"], epochs=epochs,
                              warmup_epochs=config["warmup_epochs"],
                              lambda_ramp_epochs=config["lambda_ramp_epochs"],
                              grad_clip=config["gradient_clip"], device="cpu", verbose=False),
         get_sprl_encoder_fn),
    )
    for method, init_seed, make_model, train, encoder_fn in specs:
        torch.manual_seed(init_seed)
        model = make_model().to("cpu")
        train(model)
        z = encoder_fn(model, "cpu")(s)
        e_hat, _ = estimate_propensity(z, a, method="logistic")
        rows[method] = evaluate(method, e_hat)
    return {"seed": seed, "w_u": w_u, "nominal_lambda": gamma,
            "analytic_target_value": target, "methods": rows}


def run(config: dict, first_seed: int, last_seed: int, output: Path) -> None:
    if not (1 <= first_seed <= last_seed <= 100):
        raise ValueError("Seed range must be within 1..100")
    protocol = output / "configuration.json"
    if output.is_dir() and any(output.iterdir()) and not protocol.exists():
        raise ValueError(f"Refusing nonempty output directory without a matching protocol: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if protocol.exists():
        if json.loads(protocol.read_text(encoding="utf-8")) != config:
            raise ValueError("Output directory contains a different protocol")
    else:
        protocol.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for w_u in config["w_u"]:
        condition = output / f"wu_{round(w_u * 10):02d}"
        condition.mkdir(exist_ok=True)
        for seed in range(first_seed, last_seed + 1):
            path = condition / f"seed_{seed:03d}.json"
            if path.exists():
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("seed") != seed or row.get("w_u") != w_u or list(row.get("methods", {})) != config["method_order"]:
                    raise ValueError(f"Incomplete or mismatched existing seed file: {path}")
                continue
            row = run_seed(config, w_u, seed)
            path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
            print(f"Setting A w_u={w_u} seed={seed} saved {path}", flush=True)
        completed = [json.loads(path.read_text(encoding="utf-8"))
                     for path in sorted(condition.glob("seed_???.json"))]
        raw_width = float(np.mean([row["methods"]["raw"]["width"] for row in completed]))
        summary = {
            "w_u": w_u, "n_complete_seeds": len(completed),
            "complete_paper_protocol": config["run_type"] == "paper protocol"
            and [row["seed"] for row in completed] == list(range(1, 101)),
            "methods": {
                method: {
                    "inclusion_count": sum(row["methods"][method]["includes_target"] for row in completed),
                    "mean_width": float(np.mean([row["methods"][method]["width"] for row in completed])),
                    "width_to_raw": float(np.mean([row["methods"][method]["width"]
                                                   for row in completed]) / raw_width),
                }
                for method in config["method_order"]
            },
        }
        (condition / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-seed", type=int, default=1)
    parser.add_argument("--last-seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="Use small non-paper parameters and only w_u=0.5")
    args = parser.parse_args()
    configure_cpu(args.smoke)
    last_seed = args.last_seed if args.last_seed is not None else (args.first_seed if args.smoke else 100)
    output = args.output or Path("outputs/setting_a_smoke" if args.smoke else "outputs/setting_a")
    run(load_config(args.smoke), args.first_seed, last_seed, output)


if __name__ == "__main__":
    main()
