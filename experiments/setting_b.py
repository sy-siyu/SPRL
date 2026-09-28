"""Setting B: five-step fitted action-sensitivity amplification.

Run from the release root: python -m experiments.setting_b --output outputs/setting_b
``--smoke`` is a fast non-paper execution check. CPU execution and the method
order/initialization offsets match the original Setting B implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from dgp.unified_cmdp import UnifiedCMDP, flatten_trajectories
from evaluation.gamma_z import compute_gamma_z_logistic
from representations.ae import AE, get_ae_encoder_fn, train_ae
from representations.balanced import BalancedRep, get_balanced_encoder_fn, train_balanced
from representations.contrastive import ContrastiveModel, get_contrastive_encoder_fn, train_contrastive
from representations.propensity import PropensityModel, get_propensities, train_propensity
from representations.sprl_cell import SPRL, get_sprl_encoder_fn, train_sprl
from representations.vae import VAE, get_vae_encoder_fn, train_vae
from utils.runtime import configure_cpu

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "setting_b.json"
METRICS = ("gamma_z", "gamma_z_median", "amplification", "amp_median", "delta", "u_auc", "trans_mse")


def load_config(smoke: bool = False) -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if smoke:
        config = {**config, "run_type": "SMOKE / NON-PAPER OVERRIDE",
                  "w_u": [0.5], "n_trajectories": 64, "epochs": 1,
                  "propensity_epochs": 1, "warmup_epochs": 0,
                  "lambda_ramp_epochs": 1}
    else:
        config["run_type"] = "paper protocol"
    return config


def estimate_delta(s: np.ndarray, z: np.ndarray, u: np.ndarray, n_clusters: int) -> float:
    """Original Setting B histogram diagnostic from evaluation.metrics."""
    rng = np.random.RandomState(42)
    if z.shape[0] < n_clusters:
        n_clusters = max(2, z.shape[0] // 5)
    labels = KMeans(n_clusters=n_clusters, random_state=rng.randint(10000), n_init=10).fit_predict(z)
    delta_per_cluster = []
    for c in range(n_clusters):
        mask_c = labels == c
        if mask_c.sum() < 10:
            delta_per_cluster.append(0.0)
            continue
        s_c, u_c = s[mask_c], u[mask_c]
        max_tv = 0.0
        for u_val in (0, 1):
            mask_u = u_c == u_val
            if mask_u.sum() < 5:
                continue
            s_cu = s_c[mask_u]
            tv_dims = []
            for dim in range(min(s.shape[1], 10)):
                s_range = (s_c[:, dim].min() - 0.1, s_c[:, dim].max() + 0.1)
                hist_all, edges = np.histogram(s_c[:, dim], bins=10, range=s_range, density=True)
                hist_u, _ = np.histogram(s_cu[:, dim], bins=edges, density=True)
                tv_dims.append(0.5 * np.sum(np.abs(hist_all - hist_u)) * (edges[1] - edges[0]))
            max_tv = max(max_tv, float(np.mean(tv_dims)) if tv_dims else 0.0)
        delta_per_cluster.append(max_tv)
    return float(max(delta_per_cluster)) if delta_per_cluster else 0.0


def transition_mse(model: SPRL, s: np.ndarray, a: np.ndarray, s_next: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        s_t = torch.tensor(s, dtype=torch.float32)
        a_t = torch.tensor(a, dtype=torch.float32)
        next_t = torch.tensor(s_next, dtype=torch.float32)
        z = model.encode(s_t)
        z_next = model.encode(next_t)
        predicted = model.transition_predictor(z, a_t)
        return float(torch.nn.functional.mse_loss(predicted, z_next).item())


def run_seed(config: dict, w_u: float, seed: int) -> dict:
    # DGP seed=0 is fixed across all run seeds and strengths in the source.
    dgp = UnifiedCMDP(w_u=w_u, **config["dgp"])
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    flat = flatten_trajectories(dgp.sample_trajectories(config["n_trajectories"], rng=rng))
    s, a, r, s_next, u = (flat[key] for key in ("s", "a", "r", "s_next", "u"))
    d, dz, gamma_s = dgp.state_dim, config["latent_dim"], dgp.gamma_s
    offsets = config["initialization_seed_offsets"]

    def u_auc(z: np.ndarray) -> float:
        fit = LogisticRegression(max_iter=1000, C=1.0).fit(z, u)
        return float(roc_auc_score(u, fit.predict_proba(z)[:, 1]))

    def measure(z: np.ndarray, model: SPRL | None = None) -> dict:
        gamma_z, info = compute_gamma_z_logistic(z, a, u, percentile=95, use_interactions=True)
        median = info["gamma_z_median"]
        return {"gamma_z": float(gamma_z), "gamma_z_median": float(median),
                "amplification": float(gamma_z / gamma_s), "amp_median": float(median / gamma_s),
                "delta": estimate_delta(s, z, u, config["n_delta_clusters"]),
                "u_auc": u_auc(z),
                "trans_mse": transition_mse(model, s, a, s_next) if model is not None else None}

    rows = {"raw": {"gamma_z": float(gamma_s), "gamma_z_median": float(gamma_s),
                    "amplification": 1.0, "amp_median": 1.0, "delta": 0.0,
                    "u_auc": u_auc(s), "trans_mse": None}}
    epochs = config["epochs"]
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
    )
    for method, init_seed, make_model, train, encoder_fn in specs:
        torch.manual_seed(init_seed)
        model = make_model().to("cpu")
        train(model)
        rows[method] = measure(encoder_fn(model, "cpu")(s))

    torch.manual_seed(seed + offsets["teacher"])
    teacher = PropensityModel(d).to("cpu")
    train_propensity(teacher, s, a, epochs=config["propensity_epochs"],
                     device="cpu", verbose=False)
    teacher_e = get_propensities(teacher, s, device="cpu")

    for lam in config["lambda_sweep"]:
        # Configured offsets match original: 0->4000, 5->5000, others 6000+1000*lambda.
        offset = offsets[f"lambda_{lam:g}"]
        torch.manual_seed(seed + offset)
        model = SPRL(d, dz, use_transitions=True).to("cpu")
        train_sprl(model, s, a, r, e_hat=teacher_e if lam > 0 else np.zeros(len(s)),
                   data_s_next=s_next, lambda_prop=lam,
                   sigma=config["sprl_sigma"] if lam > 0 else None,
                   epochs=epochs, warmup_epochs=config["warmup_epochs"] if lam > 0 else 0,
                   lambda_ramp_epochs=config["lambda_ramp_epochs"] if lam > 0 else 0,
                   grad_clip=config["gradient_clip"], device="cpu", verbose=False)
        rows[f"cprl_lam_{lam}"] = measure(get_sprl_encoder_fn(model, "cpu")(s), model)

    if list(rows) != config["method_order"]:
        raise ValueError("Setting B method order differs from the public configuration")
    for method, metrics in rows.items():
        if any(not np.isfinite(value) for value in metrics.values() if value is not None):
            raise ValueError(f"Nonfinite metric for {method}, w_u={w_u}, seed={seed}")
    return {"seed": seed, "w_u": w_u, "gamma_s": float(gamma_s), "methods": rows}


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
            print(f"Setting B w_u={w_u} seed={seed} saved {path}", flush=True)
        completed = [json.loads(path.read_text(encoding="utf-8"))
                     for path in sorted(condition.glob("seed_???.json"))]
        summary = {
            "w_u": w_u, "gamma_s": float(np.exp(abs(w_u))),
            "n_complete_seeds": len(completed),
            "complete_paper_protocol": config["run_type"] == "paper protocol"
            and [row["seed"] for row in completed] == list(range(1, 101)),
            "headline_lambda": config["headline_lambda"],
            "methods": {},
        }
        for method in config["method_order"]:
            summary["methods"][method] = {}
            for metric in METRICS:
                values = [row["methods"][method][metric] for row in completed
                          if row["methods"][method][metric] is not None]
                summary["methods"][method][metric] = {
                    "mean": float(np.mean(values)) if values else None,
                    "std": float(np.std(values)) if values else None,
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
    output = args.output or Path("outputs/setting_b_smoke" if args.smoke else "outputs/setting_b")
    run(load_config(args.smoke), args.first_seed, last_seed, output)


if __name__ == "__main__":
    main()
