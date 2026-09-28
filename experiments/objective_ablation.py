#!/usr/bin/env python3
"""Controlled objective ablations for SPRL.

The experiment uses the submitted Setting-B DGP and evaluation protocol, but
trains fresh encoders because the submitted checkpoints were not retained.
It separates three questions:

1. Is the observable propensity loss alone sufficient to learn a useful
   representation?
2. What changes when L_prop is added to a reward-only task objective?
3. What changes when L_prop is added to the submitted reward+transition LDM
   objective?

The clean incremental comparisons use identical encoder architectures,
initial weights, data, optimizer, and training schedule within each pair:

    reward_only              vs reward_plus_propensity(lambda)
    ldm_reward_transition    vs full_sprl(lambda=1)

The standalone propensity-only encoder has the same encoder architecture and
optimizer but no task head in its objective.  The submitted median-bandwidth
kernel makes L_prop invariant to a uniform rescaling of Z; nevertheless, the
script records latent norms and mean kernel mass as numerical-collapse checks.

Oracle U is used only in the final Gamma_Z evaluation.  Training, observable
propensity loss, and task-fidelity probes use only logged variables.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dgp.unified_cmdp import UnifiedCMDP, flatten_trajectories
from evaluation.gamma_z import compute_gamma_z_logistic
from representations.propensity import (
    PropensityModel,
    get_propensities,
    train_propensity,
)
from representations.sprl_cell import (
    SPRL,
    propensity_homogeneity_loss,
    train_sprl,
)
from utils.device import resolve_device


DEFAULT_LAMBDAS = (0.0, 0.5, 1.0, 2.0, 5.0)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def median_bandwidth(z: torch.Tensor) -> float:
    """Match the submitted median-distance bandwidth with its 0.1 floor."""
    with torch.no_grad():
        distances = torch.cdist(z, z)
        off_diagonal = ~torch.eye(
            len(z), dtype=torch.bool, device=z.device
        )
        median_distance = distances[off_diagonal].median()
        return max(float(median_distance), 0.1)


def unit_normalize(z: torch.Tensor) -> torch.Tensor:
    return F.normalize(z, p=2, dim=-1, eps=1e-8)


def train_propensity_only(
    model: SPRL,
    states: np.ndarray,
    propensity: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    grad_clip: float,
    unit_sphere: bool,
    device: str,
) -> list[dict[str, float]]:
    """Train an encoder with L_prop only.

    Dynamic median bandwidth is recomputed on every batch, exactly as in the
    submitted SPRL implementation.  ``unit_sphere`` is an optional stabilized
    variant and is never pooled with the architecture-identical comparisons.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.encoder.parameters(), lr=learning_rate
    )
    state_tensor = torch.tensor(states, dtype=torch.float32, device=device)
    propensity_tensor = torch.tensor(
        propensity, dtype=torch.float32, device=device
    )
    history: list[dict[str, float]] = []

    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(len(state_tensor), device=device)
        total_loss = 0.0
        total_norm = 0.0
        batches = 0
        for start in range(0, len(state_tensor), batch_size):
            indices = permutation[start : start + batch_size]
            z_raw = model.encoder(state_tensor[indices])
            z_used = unit_normalize(z_raw) if unit_sphere else z_raw
            sigma = median_bandwidth(z_used)
            loss = propensity_homogeneity_loss(
                z_used, propensity_tensor[indices], sigma=sigma
            )
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.encoder.parameters(), grad_clip
                )
            optimizer.step()
            total_loss += float(loss.item())
            total_norm += float(z_raw.norm(dim=1).mean().item())
            batches += 1
        history.append(
            {
                "propensity_loss": total_loss / max(batches, 1),
                "raw_latent_norm": total_norm / max(batches, 1),
            }
        )
    model.eval()
    return history


@torch.no_grad()
def encode(
    model: SPRL,
    states: np.ndarray,
    *,
    device: str,
    unit_sphere: bool = False,
    batch_size: int = 4096,
) -> np.ndarray:
    outputs = []
    model.eval()
    for start in range(0, len(states), batch_size):
        state_tensor = torch.tensor(
            states[start : start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        z = model.encode(state_tensor)
        if unit_sphere:
            z = unit_normalize(z)
        outputs.append(z.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def observable_geometry(
    z: np.ndarray,
    e_hat: np.ndarray,
    *,
    seed: int,
    max_points: int = 1024,
) -> dict[str, float]:
    """Record L_prop and kernel/norm diagnostics on a fixed subsample."""
    rng = np.random.RandomState(seed)
    if len(z) > max_points:
        indices = np.sort(
            rng.choice(len(z), size=max_points, replace=False)
        )
    else:
        indices = np.arange(len(z))
    z_tensor = torch.tensor(z[indices], dtype=torch.float32)
    e_tensor = torch.tensor(e_hat[indices], dtype=torch.float32)
    sigma = median_bandwidth(z_tensor)
    distances_squared = torch.cdist(z_tensor, z_tensor).pow(2)
    weights = torch.exp(-distances_squared / (2.0 * sigma**2))
    off_diagonal = ~torch.eye(len(z_tensor), dtype=torch.bool)
    mean_kernel_weight = float(weights[off_diagonal].mean().item())
    prop_loss = float(
        propensity_homogeneity_loss(
            z_tensor, e_tensor, sigma=sigma
        ).item()
    )

    centered = z[indices] - np.mean(z[indices], axis=0, keepdims=True)
    singular_values = np.linalg.svd(
        centered, compute_uv=False, full_matrices=False
    )
    squared = singular_values**2
    if float(np.sum(squared)) <= 1e-12:
        effective_rank = 0
    else:
        cumulative = np.cumsum(squared) / np.sum(squared)
        effective_rank = int(np.searchsorted(cumulative, 0.95) + 1)

    norms = np.linalg.norm(z[indices], axis=1)
    return {
        "n_points": int(len(indices)),
        "propensity_loss": prop_loss,
        "median_bandwidth": float(sigma),
        "mean_offdiagonal_kernel_weight": mean_kernel_weight,
        "latent_norm_mean": float(np.mean(norms)),
        "latent_norm_std": float(np.std(norms)),
        "latent_coordinate_std_mean": float(
            np.mean(np.std(z[indices], axis=0))
        ),
        "effective_rank_95": int(effective_rank),
    }


def task_fidelity_probes(
    z: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_states: np.ndarray,
    dynamics_indices: list[int],
    *,
    seed: int,
) -> dict[str, float]:
    """Use fixed ridge probes to compare task information across objectives."""
    rng = np.random.RandomState(seed + 200000)
    permutation = rng.permutation(len(z))
    split = int(0.8 * len(z))
    train_indices = permutation[:split]
    eval_indices = permutation[split:]

    z_scaler = StandardScaler().fit(z[train_indices])
    z_train = z_scaler.transform(z[train_indices])
    z_eval = z_scaler.transform(z[eval_indices])
    x_train = np.column_stack([z_train, actions[train_indices]])
    x_eval = np.column_stack([z_eval, actions[eval_indices]])

    reward_probe = Ridge(alpha=1.0)
    reward_probe.fit(x_train, rewards[train_indices])
    reward_prediction = reward_probe.predict(x_eval)
    reward_mse = mean_squared_error(
        rewards[eval_indices], reward_prediction
    )

    next_target = next_states[:, dynamics_indices]
    target_scaler = StandardScaler().fit(next_target[train_indices])
    next_train = target_scaler.transform(next_target[train_indices])
    next_eval = target_scaler.transform(next_target[eval_indices])
    transition_probe = Ridge(alpha=1.0)
    transition_probe.fit(x_train, next_train)
    transition_prediction = transition_probe.predict(x_eval)
    transition_mse = mean_squared_error(
        next_eval, transition_prediction
    )
    return {
        "probe_train_fraction": 0.8,
        "reward_linear_probe_mse": float(reward_mse),
        "next_dynamics_linear_probe_standardized_mse": float(
            transition_mse
        ),
    }


def evaluate_representation(
    z: np.ndarray,
    *,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_states: np.ndarray,
    hidden: np.ndarray,
    e_hat: np.ndarray,
    dgp: UnifiedCMDP,
    seed: int,
) -> dict[str, Any]:
    gamma_z_p95, gamma_info = compute_gamma_z_logistic(
        z,
        actions,
        hidden,
        percentile=95,
        use_interactions=True,
    )
    return {
        "gamma_z_p95": float(gamma_z_p95),
        "gamma_z_median": float(gamma_info["gamma_z_median"]),
        "amplification_p95": float(gamma_z_p95 / dgp.gamma_s),
        "amplification_median": float(
            gamma_info["gamma_z_median"] / dgp.gamma_s
        ),
        "observable": observable_geometry(z, e_hat, seed=seed),
        "task_fidelity": task_fidelity_probes(
            z,
            actions,
            rewards,
            next_states,
            dgp.idx_dynamics,
            seed=seed,
        ),
    }


def checkpoint_payload(
    model: SPRL,
    *,
    method: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "method": method,
        "seed": int(seed),
        "state_dim": int(model.encoder.net[0].in_features),
        "latent_dim": int(args.latent_dim),
        "epochs": int(args.epochs),
        "n_trajectories": int(args.n_trajectories),
        "w_u": float(args.w_u),
        "conf_strength": float(args.conf_strength),
    }


def train_task_model(
    *,
    seed_value: int,
    dgp: UnifiedCMDP,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_states: np.ndarray,
    e_hat: np.ndarray,
    lambda_prop: float,
    use_transitions: bool,
    args: argparse.Namespace,
) -> tuple[SPRL, list[dict[str, float]]]:
    torch.manual_seed(seed_value)
    model = SPRL(
        dgp.state_dim,
        args.latent_dim,
        use_transitions=use_transitions,
    ).to(args.device)
    history = train_sprl(
        model,
        states,
        actions,
        rewards,
        e_hat=e_hat if lambda_prop > 0 else np.zeros(len(states)),
        data_s_next=next_states if use_transitions else None,
        lambda_prop=lambda_prop,
        sigma=None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        warmup_epochs=args.warmup_epochs if lambda_prop > 0 else 0,
        lambda_ramp_epochs=(
            args.lambda_ramp_epochs if lambda_prop > 0 else 0
        ),
        grad_clip=args.grad_clip,
        device=args.device,
        verbose=False,
    )
    return model, history


def history_summary(
    history: list[dict[str, float]],
) -> dict[str, Any]:
    if not history:
        return {}
    return {
        "first_epoch": {
            key: float(value) for key, value in history[0].items()
        },
        "last_epoch": {
            key: float(value) for key, value in history[-1].items()
        },
    }


def run_seed(
    seed: int,
    args: argparse.Namespace,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    dgp = UnifiedCMDP(
        dim_noise=2,
        dim_instrument=2,
        dim_confounded=3,
        dim_clean=5,
        conf_strength=args.conf_strength,
        w_u=args.w_u,
        horizon=5,
        seed=0,
    )
    trajectories = dgp.sample_trajectories(
        args.n_trajectories, rng=rng
    )
    flat = flatten_trajectories(trajectories)
    states = flat["s"]
    actions = flat["a"]
    rewards = flat["r"]
    next_states = flat["s_next"]
    hidden = flat["u"]

    torch.manual_seed(seed + 9000)
    propensity_model = PropensityModel(dgp.state_dim).to(args.device)
    train_propensity(
        propensity_model,
        states,
        actions,
        epochs=args.prop_epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        device=args.device,
        verbose=False,
    )
    e_hat = get_propensities(
        propensity_model, states, device=args.device
    )
    torch.save(
        {
            "model_state_dict": propensity_model.state_dict(),
            "seed": int(seed),
            "prop_epochs": int(args.prop_epochs),
            "state_dim": int(dgp.state_dim),
        },
        checkpoint_dir / f"seed{seed:03d}_propensity.pt",
    )

    seed_result: dict[str, Any] = {
        "seed": int(seed),
        "gamma_s": float(dgp.gamma_s),
        "n_rows": int(len(states)),
        "methods": {},
    }

    # Standalone propensity-only sufficiency baseline.
    torch.manual_seed(seed + 13000)
    prop_only = SPRL(
        dgp.state_dim, args.latent_dim, use_transitions=False
    ).to(args.device)
    prop_history = train_propensity_only(
        prop_only,
        states,
        e_hat,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        grad_clip=args.grad_clip,
        unit_sphere=False,
        device=args.device,
    )
    z = encode(prop_only, states, device=args.device)
    method = "propensity_only"
    seed_result["methods"][method] = {
        **evaluate_representation(
            z,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            hidden=hidden,
            e_hat=e_hat,
            dgp=dgp,
            seed=seed,
        ),
        "training": history_summary(prop_history),
        "comparison_scope": (
            "Standalone sufficiency baseline; not pooled as an "
            "architecture-identical causal objective comparison."
        ),
    }
    torch.save(
        checkpoint_payload(
            prop_only, method=method, seed=seed, args=args
        ),
        checkpoint_dir / f"seed{seed:03d}_{method}.pt",
    )

    if args.include_unit_sphere:
        torch.manual_seed(seed + 13000)
        prop_only_unit = SPRL(
            dgp.state_dim, args.latent_dim, use_transitions=False
        ).to(args.device)
        unit_history = train_propensity_only(
            prop_only_unit,
            states,
            e_hat,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            grad_clip=args.grad_clip,
            unit_sphere=True,
            device=args.device,
        )
        z_unit = encode(
            prop_only_unit,
            states,
            device=args.device,
            unit_sphere=True,
        )
        unit_method = "propensity_only_unit_sphere"
        seed_result["methods"][unit_method] = {
            **evaluate_representation(
                z_unit,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                hidden=hidden,
                e_hat=e_hat,
                dgp=dgp,
                seed=seed,
            ),
            "training": history_summary(unit_history),
            "comparison_scope": (
                "Scale-controlled sensitivity variant only; normalization "
                "differs from the submitted architecture."
            ),
        }
        torch.save(
            checkpoint_payload(
                prop_only_unit,
                method=unit_method,
                seed=seed,
                args=args,
            ),
            checkpoint_dir / f"seed{seed:03d}_{unit_method}.pt",
        )

    # Clean reward-only versus reward+propensity family.  Every lambda starts
    # from the same model initialization.
    reward_initialization_seed = seed + 14000
    for lam in args.lambdas:
        reward_model, history = train_task_model(
            seed_value=reward_initialization_seed,
            dgp=dgp,
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            e_hat=e_hat,
            lambda_prop=lam,
            use_transitions=False,
            args=args,
        )
        method = (
            "reward_only"
            if lam == 0.0
            else f"reward_plus_propensity_lambda_{lam:g}"
        )
        z = encode(reward_model, states, device=args.device)
        seed_result["methods"][method] = {
            **evaluate_representation(
                z,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                hidden=hidden,
                e_hat=e_hat,
                dgp=dgp,
                seed=seed,
            ),
            "training": history_summary(history),
            "comparison_scope": (
                "Architecture-identical reward-only/reward+propensity "
                "comparison with common initialization."
            ),
        }
        torch.save(
            checkpoint_payload(
                reward_model, method=method, seed=seed, args=args
            ),
            checkpoint_dir / f"seed{seed:03d}_{method}.pt",
        )

    # Clean submitted-objective comparison, also with common initialization.
    full_initialization_seed = seed + 15000
    for lam, method in (
        (0.0, "ldm_reward_transition"),
        (args.full_sprl_lambda, f"full_sprl_lambda_{args.full_sprl_lambda:g}"),
    ):
        full_model, history = train_task_model(
            seed_value=full_initialization_seed,
            dgp=dgp,
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            e_hat=e_hat,
            lambda_prop=lam,
            use_transitions=True,
            args=args,
        )
        z = encode(full_model, states, device=args.device)
        seed_result["methods"][method] = {
            **evaluate_representation(
                z,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                hidden=hidden,
                e_hat=e_hat,
                dgp=dgp,
                seed=seed,
            ),
            "training": history_summary(history),
            "comparison_scope": (
                "Architecture-identical LDM/full-SPRL comparison with "
                "common initialization."
            ),
        }
        torch.save(
            checkpoint_payload(
                full_model, method=method, seed=seed, args=args
            ),
            checkpoint_dir / f"seed{seed:03d}_{method}.pt",
        )

    seed_result["elapsed_seconds"] = float(
        time.perf_counter() - started
    )
    return seed_result


def aggregate(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    methods = sorted(
        {
            method
            for result in seed_results
            for method in result["methods"]
        }
    )
    output: dict[str, Any] = {
        "n_seeds": int(len(seed_results)),
        "methods": {},
    }
    metric_paths = {
        "amplification_p95": ("amplification_p95",),
        "amplification_median": ("amplification_median",),
        "observable_propensity_loss": (
            "observable",
            "propensity_loss",
        ),
        "latent_norm_mean": ("observable", "latent_norm_mean"),
        "kernel_weight_mean": (
            "observable",
            "mean_offdiagonal_kernel_weight",
        ),
        "effective_rank_95": ("observable", "effective_rank_95"),
        "reward_probe_mse": (
            "task_fidelity",
            "reward_linear_probe_mse",
        ),
        "transition_probe_mse": (
            "task_fidelity",
            "next_dynamics_linear_probe_standardized_mse",
        ),
    }
    for method in methods:
        method_rows = [
            result["methods"][method]
            for result in seed_results
            if method in result["methods"]
        ]
        output["methods"][method] = {}
        for metric, path in metric_paths.items():
            values = []
            for row in method_rows:
                value: Any = row
                for component in path:
                    value = value[component]
                values.append(float(value))
            output["methods"][method][metric] = summarize(values)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--seed_start", type=int, default=1)
    parser.add_argument("--w_u", type=float, default=0.3)
    parser.add_argument("--conf_strength", type=float, default=2.0)
    parser.add_argument("--n_trajectories", type=int, default=5000)
    parser.add_argument("--latent_dim", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--prop_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--warmup_epochs", type=int, default=50)
    parser.add_argument("--lambda_ramp_epochs", type=int, default=50)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=list(DEFAULT_LAMBDAS),
    )
    parser.add_argument("--full_sprl_lambda", type=float, default=1.0)
    parser.add_argument("--include_unit_sphere", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output_dir",
        default="outputs/objective_ablation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    args.lambdas = tuple(float(value) for value in args.lambdas)
    torch.set_num_threads(max(1, args.torch_threads))

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "p2_objective_ablation.json"

    if args.resume and output_path.exists():
        with output_path.open() as handle:
            payload = json.load(handle)
        completed = {
            int(result["seed"]) for result in payload["seed_results"]
        }
        print(
            f"Resuming {output_path}; completed seeds={sorted(completed)}",
            flush=True,
        )
    else:
        payload = {
            "config": {
                key: value
                for key, value in vars(args).items()
                if key not in ("output_dir", "resume")
            },
            "provenance": {
                "submitted_checkpoints_used": False,
                "description": (
                    "Fresh encoders trained under the submitted Setting-B "
                    "DGP/protocol; exact submitted checkpoints were not saved."
                ),
                "oracle_u_usage": (
                    "U is used only by interaction-logistic Gamma_Z "
                    "evaluation, never by training or observable diagnostics."
                ),
            },
            "seed_results": [],
            "aggregate": None,
        }
        completed = set()

    for offset in range(args.K):
        seed = args.seed_start + offset
        if seed in completed:
            print(f"Skipping completed seed {seed}", flush=True)
            continue
        print(
            f"Running P2 ablation seed {seed} "
            f"({offset + 1}/{args.K})",
            flush=True,
        )
        result = run_seed(seed, args, checkpoint_dir)
        payload["seed_results"].append(result)
        payload["aggregate"] = aggregate(payload["seed_results"])
        atomic_json_dump(payload, output_path)
        print(
            f"  finished seed {seed} in "
            f"{result['elapsed_seconds'] / 60:.1f} min; "
            f"saved {output_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
