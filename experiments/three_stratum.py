#!/usr/bin/env python3
"""Trained stress test for the multivalued-U limit of L_prop.

This is a controlled finite-state hard-bottleneck experiment, not a rerun of
the submitted continuous Setting-B benchmark.  It uses the audited
three-category-U construction in which

    e(s) = P(A=1 | S=s) = 0.5

for every raw state, while merging a task-irrelevant state coordinate raises
Gamma from 1.966685 to 2.262273.  A two-code representation must preserve the
binary task coordinate to predict reward, so a task-aligned optimum merges the
other coordinate.  At the population propensity, L_prop is exactly zero for
every representation and therefore cannot oppose that merge.

Primary runs compare a one-step task objective with task plus L_prop at
lambda_prop in {1, 10} under the oracle e(s)=0.5, using identical
initialization and minibatch order.  A secondary branch uses an independently
estimated cell propensity.  This is not the submitted continuous SPRL
architecture: the custom two-code encoder also uses common balance and entropy
regularizers to make the hard bottleneck trainable.  Oracle Gamma values for
learned discrete maps are computed by exact enumeration over all four raw
states and all three U categories.  The categorical logistic estimator is
recorded only as a sampling diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dgp.u3_proxy_failure import U3ProxyFailureDGP
from evaluation.gamma_z import compute_gamma_z_logistic_categorical
from representations.sprl_cell import propensity_homogeneity_loss


METHODS = (
    ("task_only", "task", 0.0, "oracle"),
    ("task_plus_lprop_oracle_lambda_1", "task", 1.0, "oracle"),
    ("task_plus_lprop_oracle_lambda_10", "task", 10.0, "oracle"),
    ("task_plus_lprop_fitted_lambda_1", "task", 1.0, "fitted"),
    ("propensity_only_fitted", "propensity", 1.0, "fitted"),
)


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def odds(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=float)
    return probability / (1.0 - probability)


def state_dict_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance() -> dict[str, Any]:
    files = {
        "runner": Path(__file__).resolve(),
        "dgp": REPO_ROOT / "dgp" / "u3_proxy_failure.py",
        "evaluator": REPO_ROOT / "evaluation" / "gamma_z.py",
    }
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "files": {
            name: {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": file_sha256(path),
            }
            for name, path in files.items()
        },
    }


class TwoCodeTaskModel(nn.Module):
    """A deterministic two-cell encoder with a straight-through softmax."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 16):
        super().__init__()
        self.encoder_logits = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(
        self,
        states: torch.Tensor,
        *,
        temperature: float,
        hard: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.encoder_logits(states)
        probabilities = torch.softmax(logits / temperature, dim=-1)
        if hard:
            indices = probabilities.argmax(dim=-1)
            one_hot = F.one_hot(indices, num_classes=2).to(probabilities.dtype)
            codes = one_hot + probabilities - probabilities.detach()
        else:
            codes = probabilities
            indices = probabilities.argmax(dim=-1)
        return codes, probabilities, indices

    def predict_reward(
        self,
        codes: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat([codes, actions.reshape(-1, 1)], dim=1)
        return self.reward_head(features).squeeze(-1)


def estimate_cell_propensities(
    dgp: U3ProxyFailureDGP,
    *,
    n_samples: int,
    seed: int,
    smoothing: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate e(T,C) from an independent logged sample."""
    sample = dgp.sample(n_samples, rng=np.random.default_rng(seed))
    cell = 2 * sample["t"] + sample["c"]
    successes = np.bincount(
        cell, weights=sample["a"], minlength=4
    ).astype(float)
    counts = np.bincount(cell, minlength=4).astype(float)
    estimates = (successes + smoothing) / (counts + 2.0 * smoothing)
    return estimates, {
        "n_samples": int(n_samples),
        "counts": counts.astype(int).tolist(),
        "successes": successes.astype(int).tolist(),
        "estimates": estimates.tolist(),
        "max_abs_error_from_half": float(np.max(np.abs(estimates - 0.5))),
    }


def train_model(
    *,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    e_values: np.ndarray,
    objective: str,
    lambda_prop: float,
    seed: int,
    args: argparse.Namespace,
) -> tuple[TwoCodeTaskModel, dict[str, Any]]:
    """Train one matched-initialization objective."""
    torch.manual_seed(seed)
    model = TwoCodeTaskModel(hidden_dim=args.hidden_dim)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    states_t = torch.tensor(states, dtype=torch.float32)
    actions_t = torch.tensor(actions, dtype=torch.float32)
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    propensity_t = torch.tensor(e_values, dtype=torch.float32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1_000_003)

    history = []
    for epoch in range(args.epochs):
        progress = epoch / max(args.epochs - 1, 1)
        temperature = (
            args.temperature_start
            * (args.temperature_end / args.temperature_start) ** progress
        )
        permutation = torch.randperm(len(states_t), generator=generator)
        totals = {
            "total": 0.0,
            "reward": 0.0,
            "propensity": 0.0,
            "balance": 0.0,
            "entropy": 0.0,
        }
        batches = 0

        model.train()
        for start in range(0, len(states_t), args.batch_size):
            index = permutation[start : start + args.batch_size]
            states_b = states_t[index]
            actions_b = actions_t[index]
            rewards_b = rewards_t[index]
            propensity_b = propensity_t[index]
            codes, probabilities, _ = model.encode(
                states_b,
                temperature=temperature,
                hard=True,
            )

            reward_predictions = model.predict_reward(codes, actions_b)
            reward_loss = F.mse_loss(reward_predictions, rewards_b)
            prop_loss = propensity_homogeneity_loss(
                codes, propensity_b, sigma=args.kernel_sigma
            )
            average_code = probabilities.mean(dim=0)
            balance_loss = ((average_code - 0.5) ** 2).sum()
            entropy_loss = -(
                probabilities
                * torch.log(torch.clamp(probabilities, min=1e-8))
            ).sum(dim=1).mean()

            if objective == "task":
                primary_loss = reward_loss
            elif objective == "propensity":
                primary_loss = torch.zeros_like(reward_loss)
            else:
                raise ValueError(f"Unknown objective: {objective}")

            total_loss = (
                primary_loss
                + lambda_prop * prop_loss
                + args.balance_weight * balance_loss
                + args.entropy_weight * entropy_loss
            )
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            totals["total"] += float(total_loss.item())
            totals["reward"] += float(reward_loss.item())
            totals["propensity"] += float(prop_loss.item())
            totals["balance"] += float(balance_loss.item())
            totals["entropy"] += float(entropy_loss.item())
            batches += 1

        if epoch in (0, args.epochs - 1):
            history.append(
                {
                    "epoch": int(epoch),
                    "temperature": float(temperature),
                    **{
                        key: value / max(batches, 1)
                        for key, value in totals.items()
                    },
                }
            )

    model.eval()
    return model, {"snapshots": history}


@torch.no_grad()
def hard_codes(
    model: TwoCodeTaskModel,
    states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states_t = torch.tensor(states, dtype=torch.float32)
    codes, probabilities, indices = model.encode(
        states_t, temperature=1.0, hard=True
    )
    return (
        codes.cpu().numpy(),
        probabilities.cpu().numpy(),
        indices.cpu().numpy(),
    )


def best_binary_agreement(prediction: np.ndarray, target: np.ndarray) -> float:
    direct = float(np.mean(prediction == target))
    flipped = float(np.mean((1 - prediction) == target))
    return max(direct, flipped)


def gamma_e_for_mapping(mapping: np.ndarray, cell_e: np.ndarray) -> float:
    maximum = 1.0
    cell_odds = odds(cell_e)
    for latent in np.unique(mapping):
        values = cell_odds[mapping == latent]
        maximum = max(maximum, float(values.max() / values.min()))
    return maximum


def source_propensity_loss_for_mapping(
    prototype_codes: np.ndarray,
    cell_e: np.ndarray,
    *,
    sigma: float,
) -> float:
    """Population pair loss under the supplied four-cell propensities.

    Each raw (T,C) cell has probability one quarter.  Unlike applying the
    minibatch function to four unique prototypes, this calculation includes
    independent pairs that occupy the same raw cell in its denominator.
    """
    differences = (
        prototype_codes[:, None, :] - prototype_codes[None, :, :]
    )
    distances_squared = np.sum(differences**2, axis=-1)
    weights = np.exp(-distances_squared / (2.0 * sigma**2))
    cell_probabilities = np.full(4, 0.25)
    pair_probabilities = np.outer(cell_probabilities, cell_probabilities)
    propensity_differences = (
        cell_e[:, None] - cell_e[None, :]
    ) ** 2
    numerator = np.sum(
        pair_probabilities * weights * propensity_differences
    )
    denominator = np.sum(pair_probabilities * weights)
    return float(numerator / denominator)


def common_reward_probe(
    *,
    training_codes: np.ndarray,
    training_actions: np.ndarray,
    training_rewards: np.ndarray,
    evaluation_codes: np.ndarray,
    evaluation_actions: np.ndarray,
    evaluation_rewards: np.ndarray,
    evaluation_reward_mean: np.ndarray,
) -> dict[str, float]:
    """Evaluate task information with one common frozen-code ridge probe."""
    train_features = np.column_stack(
        [training_codes, training_actions]
    )
    evaluation_features = np.column_stack(
        [evaluation_codes, evaluation_actions]
    )
    probe = Ridge(alpha=1e-6)
    probe.fit(train_features, training_rewards)
    predictions = probe.predict(evaluation_features)
    return {
        "noisy_reward_mse": float(
            np.mean((predictions - evaluation_rewards) ** 2)
        ),
        "structural_reward_mse": float(
            np.mean((predictions - evaluation_reward_mean) ** 2)
        ),
    }


def evaluate_model(
    model: TwoCodeTaskModel,
    *,
    dgp: U3ProxyFailureDGP,
    training: dict[str, np.ndarray],
    evaluation: dict[str, np.ndarray],
    fitted_cell_e: np.ndarray,
    propensity_source: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prototypes, prototype_t, prototype_c = dgp.prototype_states(
        return_labels=True
    )
    prototype_codes, prototype_probabilities, mapping = hard_codes(
        model, prototypes
    )
    training_codes, _, _ = hard_codes(model, training["s"])
    eval_codes, _, eval_indices = hard_codes(model, evaluation["s"])

    with torch.no_grad():
        reward_predictions = model.predict_reward(
            torch.tensor(eval_codes, dtype=torch.float32),
            torch.tensor(evaluation["a"], dtype=torch.float32),
        ).cpu().numpy()
    reward_mse = float(np.mean((reward_predictions - evaluation["r"]) ** 2))
    reward_mean_mse = float(
        np.mean((reward_predictions - evaluation["r_mean"]) ** 2)
    )

    gamma_z = float(dgp.gamma_for_mapping(mapping))
    logistic_gamma, logistic_info = compute_gamma_z_logistic_categorical(
        eval_codes,
        evaluation["a"],
        evaluation["u"],
        percentile=95,
        C=args.evaluator_c,
    )
    source_cell_e = (
        np.full(4, 0.5)
        if propensity_source == "oracle"
        else fitted_cell_e
    )
    source_propensity_loss = source_propensity_loss_for_mapping(
        prototype_codes,
        source_cell_e,
        sigma=args.kernel_sigma,
    )
    probe = common_reward_probe(
        training_codes=training_codes,
        training_actions=training["a"],
        training_rewards=training["r"],
        evaluation_codes=eval_codes,
        evaluation_actions=evaluation["a"],
        evaluation_rewards=evaluation["r"],
        evaluation_reward_mean=evaluation["r_mean"],
    )

    occupancy = np.bincount(mapping, minlength=2).astype(float) / 4.0
    sample_occupancy = (
        np.bincount(eval_indices, minlength=2).astype(float)
        / len(eval_indices)
    )
    merges_c = bool(
        mapping[0] == mapping[1] and mapping[2] == mapping[3]
    )
    separates_t = bool(mapping[0] != mapping[2]) if merges_c else False
    task_aligned = bool(merges_c and separates_t)

    return {
        "prototype_mapping": mapping.astype(int).tolist(),
        "prototype_probabilities": prototype_probabilities.tolist(),
        "task_aligned_merge": task_aligned,
        "merges_c_within_t": merges_c,
        "separates_t": separates_t,
        "prototype_code_occupancy": occupancy.tolist(),
        "sample_code_occupancy": sample_occupancy.tolist(),
        "code_agreement_t": best_binary_agreement(mapping, prototype_t),
        "code_agreement_c": best_binary_agreement(mapping, prototype_c),
        "reward_mse_noisy": reward_mse,
        "reward_mse_structural_mean": reward_mean_mse,
        "gamma_z_exact": gamma_z,
        "amplification_exact": float(gamma_z / dgp.gamma_s),
        "gamma_z_logistic_p95": float(logistic_gamma),
        "amplification_logistic_p95": float(
            logistic_gamma / dgp.gamma_s
        ),
        "gamma_z_logistic_info": {
            key: value
            for key, value in logistic_info.items()
            if key
            not in {
                "category_probability_means",
                "category_intercepts",
            }
        },
        "source_propensity_loss": source_propensity_loss,
        "gamma_e_for_source_propensity": gamma_e_for_mapping(
            mapping, source_cell_e
        ),
        "common_reward_probe": probe,
        "state_dict_sha256": state_dict_hash(model),
    }


def run_seed(
    seed: int,
    *,
    dgp: U3ProxyFailureDGP,
    args: argparse.Namespace,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    training = dgp.sample(
        args.n_train,
        rng=np.random.default_rng(seed + 10_000),
        reward_noise_std=args.reward_noise_std,
    )
    evaluation = dgp.sample(
        args.n_eval,
        rng=np.random.default_rng(seed + 20_000),
        reward_noise_std=args.reward_noise_std,
    )
    fitted_cell_e, propensity_audit = estimate_cell_propensities(
        dgp,
        n_samples=args.n_propensity,
        seed=seed + 30_000,
        smoothing=args.propensity_smoothing,
    )
    training_cell = 2 * training["t"] + training["c"]
    oracle_e = np.full(args.n_train, 0.5, dtype=float)
    fitted_e = fitted_cell_e[training_cell]

    result: dict[str, Any] = {
        "seed": int(seed),
        "propensity_estimate": propensity_audit,
        "methods": {},
    }
    initialization_seed = seed + 40_000
    for method, objective, lambda_prop, source in METHODS:
        source_e = oracle_e if source == "oracle" else fitted_e
        model, training_audit = train_model(
            states=training["s"],
            actions=training["a"],
            rewards=training["r"],
            e_values=source_e,
            objective=objective,
            lambda_prop=lambda_prop,
            seed=initialization_seed,
            args=args,
        )
        method_result = evaluate_model(
            model,
            dgp=dgp,
            training=training,
            evaluation=evaluation,
            fitted_cell_e=fitted_cell_e,
            propensity_source=source,
            args=args,
        )
        method_result.update(
            {
                "objective": objective,
                "lambda_prop": float(lambda_prop),
                "propensity_source": source,
                "training": training_audit,
            }
        )
        result["methods"][method] = method_result
        torch.save(
            {
                "schema_version": 2,
                "experiment": "u3_proxy_failure_two_code_action_relevant",
                "seed": int(seed),
                "initialization_seed": int(initialization_seed),
                "method": method,
                "objective": objective,
                "lambda_prop": float(lambda_prop),
                "propensity_source": source,
                "model_state_dict": model.state_dict(),
                "config": vars(args),
                "provenance": provenance(),
                "dgp": dgp.audit_dict(),
            },
            checkpoint_dir / f"seed{seed:03d}_{method}.pt",
        )

    oracle_hashes = {
        result["methods"][name]["state_dict_sha256"]
        for name in (
            "task_only",
            "task_plus_lprop_oracle_lambda_1",
            "task_plus_lprop_oracle_lambda_10",
        )
    }
    result["oracle_matched_models_bitwise_identical"] = (
        len(oracle_hashes) == 1
    )
    result["elapsed_seconds"] = float(time.perf_counter() - started)
    return result


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def aggregate(
    seed_results: list[dict[str, Any]],
    *,
    dgp: U3ProxyFailureDGP,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "n_seeds": int(len(seed_results)),
        "methods": {},
    }
    for method, _, _, _ in METHODS:
        rows = [seed["methods"][method] for seed in seed_results]
        output["methods"][method] = {
            "task_aligned_merges": int(
                sum(bool(row["task_aligned_merge"]) for row in rows)
            ),
            "reward_mse_structural_mean": summarize(
                [float(row["reward_mse_structural_mean"]) for row in rows]
            ),
            "reward_mse_noisy": summarize(
                [float(row["reward_mse_noisy"]) for row in rows]
            ),
            "gamma_z_exact": summarize(
                [float(row["gamma_z_exact"]) for row in rows]
            ),
            "amplification_exact": summarize(
                [float(row["amplification_exact"]) for row in rows]
            ),
            "gamma_z_logistic_p95": summarize(
                [float(row["gamma_z_logistic_p95"]) for row in rows]
            ),
            "source_propensity_loss": summarize(
                [float(row["source_propensity_loss"]) for row in rows]
            ),
            "common_probe_structural_reward_mse": summarize(
                [
                    float(
                        row["common_reward_probe"][
                            "structural_reward_mse"
                        ]
                    )
                    for row in rows
                ]
            ),
            "common_probe_noisy_reward_mse": summarize(
                [
                    float(
                        row["common_reward_probe"]["noisy_reward_mse"]
                    )
                    for row in rows
                ]
            ),
            "code_agreement_t": summarize(
                [float(row["code_agreement_t"]) for row in rows]
            ),
            "code_agreement_c": summarize(
                [float(row["code_agreement_c"]) for row in rows]
            ),
        }

    primary = output["methods"]["task_plus_lprop_oracle_lambda_1"]
    output["planned_validation_gates"] = {
        "at_least_27_of_30_task_aligned": (
            len(seed_results) == 30
            and primary["task_aligned_merges"] >= 27
        ),
        "mean_common_probe_noisy_reward_mse_at_most_0.004": (
            primary["common_probe_noisy_reward_mse"]["mean"] <= 0.004
        ),
        "all_oracle_models_bitwise_identical": all(
            bool(seed["oracle_matched_models_bitwise_identical"])
            for seed in seed_results
        ),
        "target_gamma_s": float(dgp.gamma_s),
        "target_gamma_merged": float(dgp.gamma_merged),
        "target_amplification": float(dgp.amplification_merged),
    }
    return output


def write_report(payload: dict[str, Any], path: Path) -> None:
    aggregate_result = payload["aggregate"]
    dgp = payload["dgp"]
    lines = [
        "# Three-category-U trained proxy-failure stress test",
        "",
        "This is a controlled finite-state two-code hard-bottleneck objective "
        "stress test, not a reproduction of the continuous Setting B or its "
        "encoder architecture.",
        "",
        "## Population target",
        "",
        f"- Exact marginal propensities: `{dgp['marginal_propensity']}`.",
        f"- Raw `Gamma_S`: {dgp['gamma_s']:.9f}.",
        f"- Task-aligned merge `Gamma_Z`: {dgp['gamma_merged']:.9f}.",
        f"- Amplification: {dgp['amplification_merged']:.9f}x.",
        f"- Minimum hidden-cell mass: {dgp['minimum_hidden_cell_mass']:.3f}.",
        f"- Behavior-policy range: "
        f"[{dgp['minimum_policy_probability']:.3f}, "
        f"{dgp['maximum_policy_probability']:.3f}].",
        "",
        "## Trained results",
        "",
        "| Method | Task-aligned merges | Median exact amp. | "
        "Mean common-probe structural MSE | "
        "Mean source-propensity L_prop |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, _, _, _ in METHODS:
        row = aggregate_result["methods"][method]
        lines.append(
            f"| {method} | {row['task_aligned_merges']}/"
            f"{aggregate_result['n_seeds']} | "
            f"{row['amplification_exact']['median']:.6f} | "
            f"{row['common_probe_structural_reward_mse']['mean']:.6f} | "
            f"{row['source_propensity_loss']['mean']:.8f} |"
        )
    gates = aggregate_result["planned_validation_gates"]
    lines.extend(
        [
            "",
            "## Planned validation gates",
            "",
            *[
                f"- `{key}`: {value}"
                for key, value in gates.items()
            ],
            "",
            "## Interpretation boundary",
            "",
            "The oracle-propensity branches test the population objective: "
            "because every e(s)=0.5, L_prop is identically zero for every "
            "representation. The fitted branch separately records finite-"
            "sample propensity estimation behavior. The reward includes a "
            "direct action effect, but this experiment does not establish "
            "value noncoverage, estimate typical failure frequency, or test "
            "the continuous Setting-B implementation.",
            "",
            "## Reproducibility",
            "",
            *[
                f"- `{name}` SHA-256: "
                f"`{details['sha256']}`"
                for name, details in payload["provenance"]["files"].items()
            ],
            (
                f"- Python `{payload['provenance']['python'].split()[0]}`, "
                f"NumPy `{payload['provenance']['numpy']}`, "
                f"PyTorch `{payload['provenance']['torch']}`, "
                f"scikit-learn "
                f"`{payload['provenance']['scikit_learn']}`."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines))


def validate_preflight(dgp: U3ProxyFailureDGP, args: argparse.Namespace) -> None:
    assert np.allclose(dgp.marginal_propensity, 0.5, atol=1e-12)
    assert np.isclose(dgp.gamma_s, 1.966685232035, atol=1e-10)
    assert np.isclose(dgp.gamma_merged, 2.262273331555, atol=1e-10)
    assert dgp.gamma_merged > dgp.gamma_s
    assert dgp.gamma_merged <= dgp.gamma_s**2 + 1e-12
    assert dgp.minimum_hidden_cell_mass >= 0.10 - 1e-12
    assert dgp.minimum_policy_probability >= 0.38 - 1e-12
    assert dgp.maximum_policy_probability <= 0.62 + 1e-12

    arbitrary_z = torch.randn(32, 2, requires_grad=True)
    constant_e = torch.full((32,), 0.5)
    loss = propensity_homogeneity_loss(
        arbitrary_z, constant_e, sigma=args.kernel_sigma
    )
    loss.backward()
    assert float(loss.item()) == 0.0
    assert torch.count_nonzero(arbitrary_z.grad).item() == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=30)
    parser.add_argument("--seed_start", type=int, default=1)
    parser.add_argument("--n_train", type=int, default=4096)
    parser.add_argument("--n_eval", type=int, default=20_000)
    parser.add_argument("--n_propensity", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--temperature_start", type=float, default=1.0)
    parser.add_argument("--temperature_end", type=float, default=0.2)
    parser.add_argument("--balance_weight", type=float, default=0.1)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    parser.add_argument("--kernel_sigma", type=float, default=1.0)
    parser.add_argument("--reward_noise_std", type=float, default=0.05)
    parser.add_argument("--action_reward_effect", type=float, default=0.25)
    parser.add_argument("--propensity_smoothing", type=float, default=1.0)
    parser.add_argument("--evaluator_c", type=float, default=100.0)
    parser.add_argument("--torch_threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output_dir",
        default="outputs/three_stratum",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.K <= 0:
        raise ValueError("K must be positive")
    torch.set_num_threads(max(1, args.torch_threads))
    dgp = U3ProxyFailureDGP(
        reward_noise_std=args.reward_noise_std,
        action_reward_effect=args.action_reward_effect,
    )
    validate_preflight(dgp, args)

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "u3_proxy_failure_stress.json"
    report_path = output_dir / "u3_proxy_failure_stress_report.md"

    if args.resume and output_path.exists():
        with output_path.open() as handle:
            payload = json.load(handle)
        saved_config = dict(payload["config"])
        current_config = dict(vars(args))
        saved_config.pop("resume", None)
        current_config.pop("resume", None)
        if saved_config != current_config:
            raise ValueError(
                "Resume configuration differs from the saved run; use a "
                "fresh output directory instead of mixing experiments."
            )
        seed_results = payload["seed_results"]
        completed = {int(row["seed"]) for row in seed_results}
    else:
        seed_results = []
        completed = set()

    requested_seeds = range(args.seed_start, args.seed_start + args.K)
    for seed in requested_seeds:
        if seed in completed:
            continue
        result = run_seed(
            seed,
            dgp=dgp,
            args=args,
            checkpoint_dir=checkpoint_dir,
        )
        seed_results.append(result)
        seed_results.sort(key=lambda row: int(row["seed"]))
        payload = {
            "schema_version": 2,
            "experiment": "u3_proxy_failure_two_code_action_relevant",
            "config": vars(args),
            "provenance": provenance(),
            "dgp": dgp.audit_dict(),
            "seed_results": seed_results,
            "aggregate": aggregate(
                seed_results, dgp=dgp, args=args
            ),
        }
        atomic_json_dump(payload, output_path)
        write_report(payload, report_path)
        primary = result["methods"][
            "task_plus_lprop_oracle_lambda_1"
        ]
        print(
            f"seed={seed} map={primary['prototype_mapping']} "
            f"task_aligned={primary['task_aligned_merge']} "
            f"amp={primary['amplification_exact']:.6f} "
            f"mse={primary['reward_mse_structural_mean']:.6f} "
            f"elapsed={result['elapsed_seconds']:.1f}s",
            flush=True,
        )

    final_payload = {
        "schema_version": 2,
        "experiment": "u3_proxy_failure_two_code_action_relevant",
        "config": vars(args),
        "provenance": provenance(),
        "dgp": dgp.audit_dict(),
        "seed_results": seed_results,
        "aggregate": aggregate(seed_results, dgp=dgp, args=args),
    }
    atomic_json_dump(final_payload, output_path)
    write_report(final_payload, report_path)
    print(f"Wrote {output_path}", flush=True)
    print(f"Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
