#!/usr/bin/env python3
"""Independent reward-head evaluation for the stabilized P2 ablation.

For each completed training seed, this script generates a fresh Setting-B
sample with a disjoint, deterministic large RNG offset. It loads the frozen
unit-sphere reward-only and reward+propensity checkpoints, applies exactly the
same latent normalization used during training, and evaluates reward-head MSE.

The evaluation uses no hidden-U information.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dgp.unified_cmdp import UnifiedCMDP, flatten_trajectories
from representations.sprl_cell import SPRL
from utils.device import resolve_device

from experiments.objective_ablation import atomic_json_dump, unit_normalize


@torch.no_grad()
def reward_head_mse(
    model: SPRL,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> float:
    model.eval()
    squared_error_sum = 0.0
    n_rows = 0
    for start in range(0, len(states), batch_size):
        state_tensor = torch.tensor(
            states[start : start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        action_tensor = torch.tensor(
            actions[start : start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        reward_tensor = torch.tensor(
            rewards[start : start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        normalized_z = unit_normalize(model.encoder(state_tensor))
        prediction = model.reward_predictor(
            normalized_z, action_tensor
        )
        squared_error_sum += float(
            torch.sum((prediction - reward_tensor) ** 2).item()
        )
        n_rows += len(state_tensor)
    return squared_error_sum / max(n_rows, 1)


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=(
            "outputs/objective_unit_sphere/"
            "p2_stabilized_ablation.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/objective_unit_sphere/"
            "p2_stabilized_heldout_reward.json"
        ),
    )
    parser.add_argument("--n_eval_trajectories", type=int, default=1000)
    parser.add_argument("--evaluation_seed_offset", type=int, default=700000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--bootstrap_repetitions", type=int, default=20000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch_threads", type=int, default=4)
    args = parser.parse_args()
    args.device = resolve_device(args.device)
    torch.set_num_threads(max(1, args.torch_threads))

    input_path = Path(args.input)
    with input_path.open() as handle:
        training_payload = json.load(handle)
    if (
        training_payload.get("provenance", {}).get("experiment")
        != "unit_sphere_stabilized_followup"
    ):
        raise AssertionError("Input is not the stabilized P2 experiment")
    config = training_payload["config"]
    seed_results = sorted(
        training_payload["seed_results"],
        key=lambda result: int(result["seed"]),
    )
    expected_seeds = list(
        range(
            int(config["seed_start"]),
            int(config["seed_start"]) + int(config["K"]),
        )
    )
    observed_seeds = [int(result["seed"]) for result in seed_results]
    if observed_seeds != expected_seeds:
        raise AssertionError(
            f"Training run incomplete: expected {expected_seeds}, "
            f"got {observed_seeds}"
        )

    control_method = "reward_only_unit_sphere"
    treatment_method = (
        "reward_plus_propensity_unit_sphere_lambda_"
        f"{float(config['lambda_prop']):g}"
    )
    checkpoint_dir = input_path.parent / "checkpoints"
    rows: list[dict[str, Any]] = []

    for seed in observed_seeds:
        evaluation_seed = args.evaluation_seed_offset + seed
        dgp = UnifiedCMDP(
            dim_noise=2,
            dim_instrument=2,
            dim_confounded=3,
            dim_clean=5,
            conf_strength=float(config["conf_strength"]),
            w_u=float(config["w_u"]),
            horizon=5,
            seed=0,
        )
        trajectories = dgp.sample_trajectories(
            args.n_eval_trajectories,
            rng=np.random.RandomState(evaluation_seed),
        )
        flat = flatten_trajectories(trajectories)
        states = flat["s"]
        actions = flat["a"]
        rewards = flat["r"]
        reward_variance = float(np.var(rewards))
        if reward_variance <= 0:
            raise AssertionError(
                f"Seed {seed}: non-positive reward variance"
            )

        method_mse: dict[str, float] = {}
        for method in (control_method, treatment_method):
            checkpoint_path = (
                checkpoint_dir / f"seed{seed:03d}_{method}.pt"
            )
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            if checkpoint.get("unit_sphere") is not True:
                raise AssertionError(
                    f"Not a unit-sphere checkpoint: {checkpoint_path}"
                )
            if checkpoint["method"] != method:
                raise AssertionError(
                    f"Method mismatch: {checkpoint_path}"
                )
            if int(checkpoint["seed"]) != seed:
                raise AssertionError(
                    f"Seed mismatch: {checkpoint_path}"
                )
            model = SPRL(
                int(checkpoint["state_dim"]),
                int(checkpoint["latent_dim"]),
                use_transitions=False,
            ).to(args.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            method_mse[method] = reward_head_mse(
                model,
                states,
                actions,
                rewards,
                device=args.device,
                batch_size=args.batch_size,
            )

        rows.append(
            {
                "training_seed": int(seed),
                "evaluation_seed": int(evaluation_seed),
                "n_eval_trajectories": int(
                    args.n_eval_trajectories
                ),
                "n_eval_rows": int(len(states)),
                "reward_variance": reward_variance,
                "reward_head_mse": method_mse,
                "mse_over_reward_variance": {
                    method: float(value / reward_variance)
                    for method, value in method_mse.items()
                },
                "treatment_minus_control": float(
                    method_mse[treatment_method]
                    - method_mse[control_method]
                ),
            }
        )

    control = np.asarray(
        [row["reward_head_mse"][control_method] for row in rows]
    )
    treatment = np.asarray(
        [row["reward_head_mse"][treatment_method] for row in rows]
    )
    control_normalized = np.asarray(
        [
            row["mse_over_reward_variance"][control_method]
            for row in rows
        ]
    )
    treatment_normalized = np.asarray(
        [
            row["mse_over_reward_variance"][treatment_method]
            for row in rows
        ]
    )
    difference = treatment - control
    rng = np.random.RandomState(20260724)
    indices = rng.randint(
        0,
        len(difference),
        size=(args.bootstrap_repetitions, len(difference)),
    )
    bootstrap_means = difference[indices].mean(axis=1)
    ci_low, ci_high = np.percentile(
        bootstrap_means, [2.5, 97.5]
    )
    output = {
        "source_training_result": str(input_path),
        "evaluation": {
            "dgp": "submitted Setting B",
            "n_eval_trajectories_per_training_seed": int(
                args.n_eval_trajectories
            ),
            "n_eval_rows_per_training_seed": int(
                args.n_eval_trajectories * 5
            ),
            "evaluation_seed_rule": (
                "evaluation_seed = training_seed + "
                f"{args.evaluation_seed_offset}"
            ),
            "frozen_checkpoints": True,
            "unit_normalization_applied": True,
            "oracle_u_used": False,
        },
        "control": control_method,
        "treatment": treatment_method,
        "seed_results": rows,
        "aggregate": {
            control_method: summarize(control),
            treatment_method: summarize(treatment),
            "mse_over_reward_variance": {
                control_method: summarize(control_normalized),
                treatment_method: summarize(treatment_normalized),
            },
            "treatment_minus_control": {
                **summarize(difference),
                "negative_is_favorable": True,
                "treatment_better_seed_count": int(
                    np.sum(difference < 0)
                ),
                "ties": int(np.sum(difference == 0)),
                "bootstrap_mean_95ci": [
                    float(ci_low),
                    float(ci_high),
                ],
                "bootstrap_repetitions": int(
                    args.bootstrap_repetitions
                ),
                "bootstrap_seed": 20260724,
            },
        },
    }
    atomic_json_dump(output, Path(args.output))
    print(
        f"Evaluated {len(rows)} frozen checkpoint pairs; "
        f"wrote {args.output}"
    )


if __name__ == "__main__":
    main()
