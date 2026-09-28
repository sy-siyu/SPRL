#!/usr/bin/env python3
"""Paired latent-dimension sensitivity for SPRL.

This script varies only ``d_Z`` over a pre-specified grid and compares the
submitted Setting-B objectives:

    LDM:  reward + transition
    SPRL: reward + transition + lambda_prop * L_prop

Within each (seed, d_Z) pair, both models use identical data, architecture,
initial weights, and minibatch RNG through the warmup. The propensity model is
shared. Oracle U is used only in the final Gamma_Z evaluation.

The lambda=1 choice is inherited from the submitted oracle-evaluated sweep; it
is not retuned for this experiment. All encoders are fresh rebuttal runs.
Results and checkpoints are written incrementally so ``--resume`` is safe.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dgp.unified_cmdp import UnifiedCMDP, flatten_trajectories
from representations.propensity import (
    PropensityModel,
    get_propensities,
    train_propensity,
)
from representations.sprl_cell import SPRL, train_sprl
from experiments.objective_ablation import (
    encode,
    evaluate_representation,
    history_summary,
)
from utils.device import resolve_device


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def checkpoint_payload(
    model: SPRL,
    *,
    method: str,
    seed: int,
    latent_dim: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "method": method,
        "seed": int(seed),
        "latent_dim": int(latent_dim),
        "epochs": int(args.epochs),
        "n_trajectories": int(args.n_trajectories),
        "w_u": float(args.w_u),
        "lambda_prop": 0.0 if method == "ldm" else float(args.lambda_prop),
        "fresh_rebuttal_run": True,
    }


def train_from_common_initialization(
    *,
    initialization_seed: int,
    state_dim: int,
    latent_dim: int,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_states: np.ndarray,
    e_hat: np.ndarray,
    lambda_prop: float,
    args: argparse.Namespace,
) -> tuple[SPRL, list[dict[str, float]]]:
    # This exactly matches ``train_task_model`` in the existing P2 run:
    # resetting before construction gives both methods identical parameters
    # and leaves the RNG at the same post-construction state before training.
    torch.manual_seed(initialization_seed)
    model = SPRL(
        state_dim, latent_dim, use_transitions=True
    ).to(args.device)
    history = train_sprl(
        model,
        states,
        actions,
        rewards,
        e_hat=e_hat if lambda_prop > 0 else np.zeros(len(states)),
        data_s_next=next_states,
        lambda_prop=lambda_prop,
        sigma=None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        warmup_epochs=args.warmup_epochs if lambda_prop > 0 else 0,
        lambda_ramp_epochs=args.lambda_ramp_epochs if lambda_prop > 0 else 0,
        grad_clip=args.grad_clip,
        device=args.device,
        verbose=False,
    )
    return model, history


def run_pair(
    *,
    latent_dim: int,
    seed: int,
    args: argparse.Namespace,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    rng = np.random.RandomState(seed)
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
    flat = flatten_trajectories(
        dgp.sample_trajectories(args.n_trajectories, rng=rng)
    )
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
    e_hat = get_propensities(propensity_model, states, device=args.device)
    torch.save(
        {
            "model_state_dict": propensity_model.state_dict(),
            "seed": int(seed),
            "latent_dim_scope": int(latent_dim),
            "prop_epochs": int(args.prop_epochs),
        },
        checkpoint_dir / f"dz{latent_dim}_seed{seed:03d}_propensity.pt",
    )

    initialization_seed = seed + 15000
    methods: dict[str, Any] = {}
    warmup_histories: dict[str, list[dict[str, float]]] = {}
    for method, lam in (("ldm", 0.0), ("sprl_lambda_1", args.lambda_prop)):
        model, history = train_from_common_initialization(
            initialization_seed=initialization_seed,
            state_dim=dgp.state_dim,
            latent_dim=latent_dim,
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            e_hat=e_hat,
            lambda_prop=lam,
            args=args,
        )
        warmup_histories[method] = history
        z = encode(model, states, device=args.device)
        methods[method] = {
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
        }
        torch.save(
            checkpoint_payload(
                model,
                method=method,
                seed=seed,
                latent_dim=latent_dim,
                args=args,
            ),
            checkpoint_dir / f"dz{latent_dim}_seed{seed:03d}_{method}.pt",
        )

    warmup_epochs = min(args.warmup_epochs, len(warmup_histories["ldm"]))
    maximum_difference = 0.0
    for epoch in range(warmup_epochs):
        for metric in ("reward", "transition"):
            difference = abs(
                warmup_histories["ldm"][epoch][metric]
                - warmup_histories["sprl_lambda_1"][epoch][metric]
            )
            maximum_difference = max(maximum_difference, difference)

    return {
        "latent_dim": int(latent_dim),
        "seed": int(seed),
        "gamma_s": float(dgp.gamma_s),
        "n_rows": int(len(states)),
        "methods": methods,
        "paired_warmup_audit": {
            "epochs_compared": int(warmup_epochs),
            "maximum_task_loss_difference": float(maximum_difference),
            "passed": bool(maximum_difference <= 1e-10),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for latent_dim in sorted({int(row["latent_dim"]) for row in rows}):
        selected = [
            row for row in rows if int(row["latent_dim"]) == latent_dim
        ]
        dimension_summary: dict[str, Any] = {
            "n_seeds": len(selected),
            "methods": {},
            "paired_differences": {},
        }
        paths = {
            "amplification_p95": ("amplification_p95",),
            "amplification_median": ("amplification_median",),
            "propensity_loss": ("observable", "propensity_loss"),
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
        for method in ("ldm", "sprl_lambda_1"):
            dimension_summary["methods"][method] = {}
            for metric, path in paths.items():
                values = []
                for row in selected:
                    value: Any = row["methods"][method]
                    for component in path:
                        value = value[component]
                    values.append(float(value))
                dimension_summary["methods"][method][metric] = summarize(
                    values
                )
        for metric, path in paths.items():
            differences = []
            for row in selected:
                ldm_value: Any = row["methods"]["ldm"]
                sprl_value: Any = row["methods"]["sprl_lambda_1"]
                for component in path:
                    ldm_value = ldm_value[component]
                    sprl_value = sprl_value[component]
                differences.append(float(sprl_value) - float(ldm_value))
            dimension_summary["paired_differences"][
                f"sprl_minus_ldm_{metric}"
            ] = summarize(differences)
        output[str(latent_dim)] = dimension_summary
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dims", type=int, nargs="+", default=[2, 10])
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--seed_start", type=int, default=1)
    parser.add_argument("--w_u", type=float, default=0.3)
    parser.add_argument("--conf_strength", type=float, default=2.0)
    parser.add_argument("--n_trajectories", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--prop_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--warmup_epochs", type=int, default=50)
    parser.add_argument("--lambda_ramp_epochs", type=int, default=50)
    parser.add_argument("--lambda_prop", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output_dir",
        default="outputs/latent_dimension",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    torch.set_num_threads(max(1, args.torch_threads))
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "latent_dimension_sensitivity.json"

    if args.resume and output_path.exists():
        with output_path.open() as handle:
            payload = json.load(handle)
        prior_config = payload["config"]
        immutable_keys = (
            "seed_start",
            "w_u",
            "conf_strength",
            "n_trajectories",
            "epochs",
            "prop_epochs",
            "batch_size",
            "learning_rate",
            "warmup_epochs",
            "lambda_ramp_epochs",
            "lambda_prop",
            "grad_clip",
        )
        for key in immutable_keys:
            if prior_config[key] != getattr(args, key):
                raise ValueError(
                    f"Cannot resume with changed {key}: "
                    f"{prior_config[key]} != {getattr(args, key)}"
                )
        prior_config["K"] = max(int(prior_config["K"]), int(args.K))
        prior_config["latent_dims"] = sorted(
            {
                *(int(value) for value in prior_config["latent_dims"]),
                *(int(value) for value in args.latent_dims),
            }
        )
    else:
        payload = {
            "config": {
                key: value
                for key, value in vars(args).items()
                if key not in ("output_dir", "resume")
            },
            "provenance": {
                "fresh_rebuttal_runs": True,
                "submitted_checkpoints_used": False,
                "submitted_dgp_and_hyperparameters": True,
                "lambda_selection": (
                    "lambda_prop=1 is inherited from the submitted "
                    "oracle-evaluated sweep and is not retuned here"
                ),
                "oracle_u_usage": (
                    "U is used only by interaction-logistic Gamma_Z "
                    "evaluation, never by training or observable diagnostics"
                ),
                "dimension_5_source": (
                    "Existing exact-protocol K=10 P2 run can be summarized "
                    "separately; this file contains only dimensions executed "
                    "through this script"
                ),
            },
            "seed_results": [],
            "aggregate": {},
        }

    completed = {
        (int(row["latent_dim"]), int(row["seed"]))
        for row in payload["seed_results"]
    }
    for latent_dim in args.latent_dims:
        for offset in range(args.K):
            seed = args.seed_start + offset
            if (latent_dim, seed) in completed:
                continue
            row = run_pair(
                latent_dim=latent_dim,
                seed=seed,
                args=args,
                checkpoint_dir=checkpoint_dir,
            )
            payload["seed_results"].append(row)
            payload["aggregate"] = aggregate(payload["seed_results"])
            atomic_json_dump(payload, output_path)
            print(
                f"d_Z={latent_dim} seed={seed}: "
                f"LDM={row['methods']['ldm']['amplification_p95']:.3f}, "
                f"SPRL={row['methods']['sprl_lambda_1']['amplification_p95']:.3f}, "
                f"elapsed={row['elapsed_seconds']:.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
