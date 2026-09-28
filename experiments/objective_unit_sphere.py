#!/usr/bin/env python3
"""Minimal unit-sphere follow-up to the P2 objective ablation.

The exact submitted implementation recomputes a median kernel bandwidth and
then detaches it as a scalar.  Although the recomputed objective value is
invariant to uniform rescaling, its per-step gradient treats the bandwidth as
fixed and can drive the raw latent radius outward.  This follow-up separates
that numerical/geometric pathology from objective sufficiency.

It is a new stabilized variant, not the submitted SPRL implementation.  Only
three representations are trained:

    propensity_only_unit_sphere
    reward_only_unit_sphere
    reward_plus_propensity_unit_sphere_lambda_1

The reward pair has identical initialization, architecture, data, optimizer,
and training schedule.  Unit normalization is used inside both the reward and
propensity objectives and again at evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dgp.unified_cmdp import UnifiedCMDP, flatten_trajectories
from representations.propensity import PropensityModel, get_propensities
from representations.sprl_cell import (
    SPRL,
    propensity_homogeneity_loss,
)
from utils.device import resolve_device

from experiments.objective_ablation import (
    aggregate,
    atomic_json_dump,
    checkpoint_payload,
    encode,
    evaluate_representation,
    history_summary,
    median_bandwidth,
    train_propensity_only,
    unit_normalize,
)


def stabilized_checkpoint_payload(
    model: SPRL,
    *,
    method: str,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = checkpoint_payload(
        model, method=method, seed=seed, args=args
    )
    payload.update(
        {
            "unit_sphere": True,
            "normalization_location": (
                "inside_training_objectives_and_at_evaluation"
            ),
            "submitted_method": False,
        }
    )
    return payload


def raw_and_angular_geometry(
    raw_z: np.ndarray,
    normalized_z: np.ndarray,
    *,
    seed: int,
    max_points: int = 1024,
) -> dict[str, Any]:
    """Diagnose conditioning hidden by the tautological unit norm."""
    rng = np.random.RandomState(seed + 310000)
    if len(raw_z) > max_points:
        indices = np.sort(
            rng.choice(len(raw_z), size=max_points, replace=False)
        )
    else:
        indices = np.arange(len(raw_z))
    raw_norms = np.linalg.norm(raw_z[indices], axis=1)
    cosine_similarity = (
        normalized_z[indices] @ normalized_z[indices].T
    )
    cosine_distance = np.clip(1.0 - cosine_similarity, 0.0, 2.0)
    off_diagonal = ~np.eye(len(indices), dtype=bool)
    angular_values = cosine_distance[off_diagonal]
    return {
        "n_points": int(len(indices)),
        "raw_latent_norm": {
            "min": float(np.min(raw_norms)),
            "p05": float(np.percentile(raw_norms, 5)),
            "median": float(np.median(raw_norms)),
            "mean": float(np.mean(raw_norms)),
            "p95": float(np.percentile(raw_norms, 95)),
            "max": float(np.max(raw_norms)),
            "fraction_below_1e-6": float(np.mean(raw_norms < 1e-6)),
        },
        "unit_sphere_cosine_distance": {
            "p05": float(np.percentile(angular_values, 5)),
            "median": float(np.median(angular_values)),
            "mean": float(np.mean(angular_values)),
            "p95": float(np.percentile(angular_values, 95)),
        },
    }


@torch.no_grad()
def in_sample_reward_head_mse(
    model: SPRL,
    normalized_z: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    *,
    device: str,
    batch_size: int = 4096,
) -> float:
    """Evaluate the trained head on all training rows (descriptive only)."""
    model.eval()
    predictions = []
    for start in range(0, len(normalized_z), batch_size):
        z_tensor = torch.tensor(
            normalized_z[start : start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        action_tensor = torch.tensor(
            actions[start : start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        predictions.append(
            model.reward_predictor(z_tensor, action_tensor)
            .cpu()
            .numpy()
        )
    prediction = np.concatenate(predictions)
    return float(np.mean((prediction - rewards) ** 2))


def train_unit_sphere_reward_model(
    *,
    seed_value: int,
    dgp: UnifiedCMDP,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    e_hat: np.ndarray,
    lambda_prop: float,
    args: argparse.Namespace,
) -> tuple[SPRL, list[dict[str, float]]]:
    """Train reward or reward+propensity using normalized Z throughout."""
    torch.manual_seed(seed_value)
    model = SPRL(
        dgp.state_dim,
        args.latent_dim,
        use_transitions=False,
    ).to(args.device)
    optimizer = torch.optim.Adam(
        list(model.encoder.parameters())
        + list(model.reward_predictor.parameters()),
        lr=args.learning_rate,
    )
    states_tensor = torch.tensor(
        states, dtype=torch.float32, device=args.device
    )
    actions_tensor = torch.tensor(
        actions, dtype=torch.float32, device=args.device
    )
    rewards_tensor = torch.tensor(
        rewards, dtype=torch.float32, device=args.device
    )
    propensity_tensor = torch.tensor(
        e_hat, dtype=torch.float32, device=args.device
    )
    history: list[dict[str, float]] = []

    for epoch in range(args.epochs):
        model.train()
        permutation = torch.randperm(
            len(states_tensor), device=args.device
        )
        if lambda_prop <= 0 or epoch < args.warmup_epochs:
            lambda_effective = 0.0
        elif args.lambda_ramp_epochs > 0:
            advanced_epoch = epoch - args.warmup_epochs
            ramp = min(
                1.0,
                (advanced_epoch + 1) / args.lambda_ramp_epochs,
            )
            lambda_effective = lambda_prop * ramp
        else:
            lambda_effective = lambda_prop

        totals = {
            "total": 0.0,
            "reward": 0.0,
            "transition": 0.0,
            "prop": 0.0,
            "raw_latent_norm": 0.0,
            "normalized_latent_norm": 0.0,
        }
        batches = 0
        for start in range(0, len(states_tensor), args.batch_size):
            indices = permutation[start : start + args.batch_size]
            raw_z = model.encoder(states_tensor[indices])
            z = unit_normalize(raw_z)
            reward_prediction = model.reward_predictor(
                z, actions_tensor[indices]
            )
            reward_loss = F.mse_loss(
                reward_prediction, rewards_tensor[indices]
            )
            if lambda_effective > 0:
                sigma = median_bandwidth(z)
                prop_loss = propensity_homogeneity_loss(
                    z, propensity_tensor[indices], sigma=sigma
                )
            else:
                prop_loss = torch.tensor(0.0, device=args.device)
            total_loss = (
                reward_loss + lambda_effective * prop_loss
            )
            optimizer.zero_grad()
            total_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
            optimizer.step()

            totals["total"] += float(total_loss.item())
            totals["reward"] += float(reward_loss.item())
            totals["prop"] += float(prop_loss.item())
            totals["raw_latent_norm"] += float(
                raw_z.norm(dim=1).mean().item()
            )
            totals["normalized_latent_norm"] += float(
                z.norm(dim=1).mean().item()
            )
            batches += 1
        history.append(
            {
                key: value / max(batches, 1)
                for key, value in totals.items()
            }
        )
    model.eval()
    return model, history


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

    source_checkpoint = (
        Path(args.source_output_dir)
        / "checkpoints"
        / f"seed{seed:03d}_propensity.pt"
    )
    if not source_checkpoint.exists():
        raise FileNotFoundError(
            f"Run the exact-protocol ablation first; missing "
            f"{source_checkpoint}"
        )
    propensity_payload = torch.load(
        source_checkpoint, map_location="cpu", weights_only=False
    )
    if int(propensity_payload["seed"]) != seed:
        raise AssertionError(
            f"Propensity checkpoint seed mismatch: {source_checkpoint}"
        )
    if int(propensity_payload["state_dim"]) != dgp.state_dim:
        raise AssertionError(
            f"Propensity checkpoint dimension mismatch: {source_checkpoint}"
        )
    propensity_model = PropensityModel(dgp.state_dim).to(args.device)
    propensity_model.load_state_dict(
        propensity_payload["model_state_dict"]
    )
    e_hat = get_propensities(
        propensity_model, states, device=args.device
    )

    seed_result: dict[str, Any] = {
        "seed": int(seed),
        "gamma_s": float(dgp.gamma_s),
        "n_rows": int(len(states)),
        "source_propensity_checkpoint": str(source_checkpoint),
        "methods": {},
    }

    # Stabilized propensity-only sufficiency baseline.
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
        unit_sphere=True,
        device=args.device,
    )
    raw_z = encode(
        prop_only,
        states,
        device=args.device,
        unit_sphere=False,
    )
    z = encode(
        prop_only,
        states,
        device=args.device,
        unit_sphere=True,
    )
    method = "propensity_only_unit_sphere"
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
        "stabilization_geometry": raw_and_angular_geometry(
            raw_z, z, seed=seed
        ),
        "comparison_scope": (
            "Stabilized propensity-only sufficiency baseline; "
            "not the submitted representation geometry."
        ),
        "normalization": "unit_sphere_in_objective_and_evaluation",
    }
    torch.save(
        stabilized_checkpoint_payload(
            prop_only, method=method, seed=seed, args=args
        ),
        checkpoint_dir / f"seed{seed:03d}_{method}.pt",
    )

    # Stabilized reward-only versus reward+propensity pair. Both models start
    # from the same initialization, which also matches the raw reward family.
    reward_initialization_seed = seed + 14000
    reward_histories: dict[str, list[dict[str, float]]] = {}
    for lambda_prop, method in (
        (0.0, "reward_only_unit_sphere"),
        (
            args.lambda_prop,
            f"reward_plus_propensity_unit_sphere_lambda_"
            f"{args.lambda_prop:g}",
        ),
    ):
        model, history = train_unit_sphere_reward_model(
            seed_value=reward_initialization_seed,
            dgp=dgp,
            states=states,
            actions=actions,
            rewards=rewards,
            e_hat=e_hat,
            lambda_prop=lambda_prop,
            args=args,
        )
        reward_histories[method] = history
        raw_z = encode(
            model,
            states,
            device=args.device,
            unit_sphere=False,
        )
        z = encode(
            model,
            states,
            device=args.device,
            unit_sphere=True,
        )
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
            "stabilization_geometry": raw_and_angular_geometry(
                raw_z, z, seed=seed
            ),
            "trained_head": {
                "in_sample_reward_mse": in_sample_reward_head_mse(
                    model,
                    z,
                    actions,
                    rewards,
                    device=args.device,
                ),
                "scope": (
                    "Descriptive full-training-data fit; not held-out "
                    "generalization."
                ),
            },
            "comparison_scope": (
                "Unit-sphere reward/reward+propensity pair with common "
                "initialization; stabilized follow-up, not submitted SPRL."
            ),
            "normalization": "unit_sphere_in_objective_and_evaluation",
        }
        torch.save(
            stabilized_checkpoint_payload(
                model, method=method, seed=seed, args=args
            ),
            checkpoint_dir / f"seed{seed:03d}_{method}.pt",
        )

    control_history = reward_histories["reward_only_unit_sphere"]
    treatment_name = (
        "reward_plus_propensity_unit_sphere_lambda_"
        f"{args.lambda_prop:g}"
    )
    treatment_history = reward_histories[treatment_name]
    warmup_differences = [
        abs(
            control_history[epoch]["reward"]
            - treatment_history[epoch]["reward"]
        )
        for epoch in range(min(args.warmup_epochs, args.epochs))
    ]
    maximum_warmup_difference = max(warmup_differences, default=0.0)
    if maximum_warmup_difference > 1e-10:
        raise AssertionError(
            "Paired unit-sphere arms diverged before Lprop activation: "
            f"max reward-loss difference={maximum_warmup_difference}"
        )
    seed_result["paired_warmup_audit"] = {
        "epochs_compared": int(
            min(args.warmup_epochs, args.epochs)
        ),
        "maximum_reward_loss_difference": float(
            maximum_warmup_difference
        ),
        "passed": True,
        "meaning": (
            "Common initialization and minibatch RNG yielded identical "
            "training before the propensity penalty activated."
        ),
    }

    seed_result["elapsed_seconds"] = float(
        time.perf_counter() - started
    )
    return seed_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--seed_start", type=int, default=1)
    parser.add_argument("--w_u", type=float, default=0.3)
    parser.add_argument("--conf_strength", type=float, default=2.0)
    parser.add_argument("--n_trajectories", type=int, default=5000)
    parser.add_argument("--latent_dim", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--warmup_epochs", type=int, default=50)
    parser.add_argument("--lambda_ramp_epochs", type=int, default=50)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lambda_prop", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--source_output_dir",
        default="outputs/objective_ablation",
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "outputs/objective_unit_sphere"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    torch.set_num_threads(max(1, args.torch_threads))

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "p2_stabilized_ablation.json"

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
                "experiment": "unit_sphere_stabilized_followup",
                "submitted_method": False,
                "submitted_checkpoints_used": False,
                "description": (
                    "Fresh stabilized encoders; propensity estimates are "
                    "reused from the fresh exact-protocol P2 run."
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
            f"Running stabilized P2 seed {seed} "
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
