#!/usr/bin/env python3
"""Observable propensity-range diagnostics for learned SPRL representations.

This experiment has two distinct goals:

1. Examine a Monte Carlo analogue of the binary-U bound after discretizing a learned
   representation with k-means:

       Gamma_C <= Gamma_S * Gamma_e(C),

   where C = q(phi(S)) is the discrete representation and Gamma_e(C) is the
   within-cell odds range of e(S) = P(A=1 | S).

2. Evaluate a practical k-nearest-neighbour propensity-range score on the
   original continuous representation.  This local score is observable from
   logged (S,A), but it is a diagnostic rather than a formal bound for the
   continuous representation.

The submitted result files do not contain model checkpoints or embeddings, so
the script reproducibly retrains only the Setting-B LDM/SPRL lambda sweep and
saves the resulting checkpoints.  Seed offsets and training defaults match
the Setting B experiment.
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
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dgp.unified_cmdp import UnifiedCMDP, flatten_trajectories
from evaluation.gamma_z import compute_gamma_z_logistic
from experiments.teacher_noise import compute_transition_mse
from representations.propensity import (
    PropensityModel,
    get_propensities,
    train_propensity,
)
from representations.sprl_cell import SPRL, get_sprl_encoder_fn, train_sprl
from utils.device import resolve_device


DEFAULT_LAMBDAS = (0.0, 0.5, 1.0, 2.0, 5.0)


def odds(probability: np.ndarray, clip: float = 1e-8) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), clip, 1.0 - clip)
    return p / (1.0 - p)


def summarize(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
    }


def local_gamma_e_diagnostic(
    z: np.ndarray,
    e_hat: np.ndarray,
    *,
    k_values: tuple[int, ...] = (25, 50, 100),
    clip: float = 0.01,
    max_points: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute observable local propensity-odds ranges in Z-space.

    For each evaluation point, find its k nearest neighbours and compute both
    the full sample odds range and a 95%/5% robust odds range.  These are
    post-hoc risk scores for a continuous representation, not finite-sample
    coverage certificates.
    """
    z = np.atleast_2d(np.asarray(z, dtype=float))
    e_hat = np.asarray(e_hat, dtype=float).ravel()
    if len(z) != len(e_hat):
        raise ValueError("z and e_hat must have equal length")

    rng = np.random.RandomState(seed)
    if len(z) > max_points:
        eval_idx = np.sort(rng.choice(len(z), size=max_points, replace=False))
    else:
        eval_idx = np.arange(len(z))

    z_scaled = StandardScaler().fit_transform(z)
    max_k = min(max(k_values), len(z))
    nn = NearestNeighbors(n_neighbors=max_k, algorithm="auto", n_jobs=-1)
    nn.fit(z_scaled)
    neighbour_idx = nn.kneighbors(
        z_scaled[eval_idx], n_neighbors=max_k, return_distance=False
    )
    e_odds = odds(e_hat, clip=clip)

    result: dict[str, Any] = {
        "n_reference": int(len(z)),
        "n_queries": int(len(eval_idx)),
        "clip": float(clip),
        "note": "Continuous-Z kNN risk score; not a formal Gamma_Z bound.",
    }
    for requested_k in k_values:
        k = min(requested_k, max_k)
        neighbourhood_odds = e_odds[neighbour_idx[:, :k]]
        full_ratio = (
            neighbourhood_odds.max(axis=1)
            / np.maximum(neighbourhood_odds.min(axis=1), 1e-12)
        )
        robust_ratio = (
            np.percentile(neighbourhood_odds, 95, axis=1)
            / np.maximum(np.percentile(neighbourhood_odds, 5, axis=1), 1e-12)
        )
        result[str(requested_k)] = {
            "effective_k": int(k),
            "full_range": summarize(full_ratio),
            "q95_q05_range": summarize(robust_ratio),
        }
    return result


def _range_ratio(
    values: np.ndarray, *, robust: bool = False, clip: float = 1e-8
) -> float:
    value_odds = odds(values, clip=clip)
    if robust:
        upper = np.percentile(value_odds, 95)
        lower = np.percentile(value_odds, 5)
    else:
        upper = np.max(value_odds)
        lower = np.min(value_odds)
    return float(upper / max(lower, 1e-12))


def discrete_partition_diagnostic(
    z_fit: np.ndarray,
    z_eval: np.ndarray,
    s: np.ndarray,
    u: np.ndarray,
    e_hat: np.ndarray,
    e_true: np.ndarray,
    dgp: UnifiedCMDP,
    *,
    n_clusters: int = 10,
    min_u_count: int = 50,
    diagnostic_clip: float = 0.02,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate Gamma_e and Gamma_Z after deterministic k-means quantization.

    The theorem applies to the population discrete representation C=q(Z) with
    the true e(S).  Values here are Monte Carlo approximations on an
    independent evaluation sample.  The e_hat version remains affected by
    propensity-estimation error.
    """
    z_fit = np.atleast_2d(np.asarray(z_fit, dtype=float))
    z_eval = np.atleast_2d(np.asarray(z_eval, dtype=float))
    s = np.atleast_2d(np.asarray(s, dtype=float))
    u = np.asarray(u).ravel().astype(int)
    e_hat = np.asarray(e_hat, dtype=float).ravel()
    e_true = np.asarray(e_true, dtype=float).ravel()

    scaler = StandardScaler().fit(z_fit)
    z_fit_scaled = scaler.transform(z_fit)
    z_eval_scaled = scaler.transform(z_eval)
    quantizer = KMeans(
        n_clusters=n_clusters, random_state=seed, n_init=10
    ).fit(z_fit_scaled)
    labels = quantizer.predict(z_eval_scaled)

    per_cell: list[dict[str, Any]] = []
    for cell in range(n_clusters):
        mask = labels == cell
        counts = [int(np.sum(mask & (u == value))) for value in (0, 1)]
        if min(counts) < min_u_count:
            continue

        # Monte Carlo approximation to pi_b(1 | C=cell,U=u): average the
        # known behavior policy over states sampled from S | C,U.
        latent_policy = []
        for value in (0, 1):
            group = mask & (u == value)
            latent_policy.append(float(np.mean(dgp.pi_b(s[group], value))))
        latent_odds = odds(np.asarray(latent_policy), clip=1e-8)
        gamma_z_cell = float(
            max(latent_odds[1] / latent_odds[0], latent_odds[0] / latent_odds[1])
        )

        per_cell.append(
            {
                "cell": int(cell),
                "count": int(np.sum(mask)),
                "u_counts": counts,
                "gamma_z_oracle_mc": gamma_z_cell,
                "gamma_e_true_full": _range_ratio(
                    e_true[mask], robust=False, clip=1e-8
                ),
                "gamma_e_true_q95_q05": _range_ratio(
                    e_true[mask], robust=True, clip=1e-8
                ),
                "gamma_e_hat_full": _range_ratio(
                    e_hat[mask], robust=False, clip=diagnostic_clip
                ),
                "gamma_e_hat_q95_q05": _range_ratio(
                    e_hat[mask], robust=True, clip=diagnostic_clip
                ),
            }
        )

    if not per_cell:
        return {
            "available": False,
            "n_clusters_requested": int(n_clusters),
            "n_cells_used": 0,
            "min_u_count": int(min_u_count),
            "diagnostic_propensity_clip": float(diagnostic_clip),
            "gamma_z_oracle_mc": None,
            "gamma_e_true_full": None,
            "gamma_e_true_q95_q05": None,
            "gamma_e_hat_full": None,
            "gamma_e_hat_q95_q05": None,
            "gamma_s_times_gamma_e_true": None,
            "gamma_s_times_gamma_e_hat": None,
            "oracle_bound_holds_mc": None,
            "cells": [],
            "note": (
                "Formal partition check unavailable: no evaluation cell met "
                "the minimum support under both U values."
            ),
        }

    gamma_z = max(row["gamma_z_oracle_mc"] for row in per_cell)
    gamma_e_true = max(row["gamma_e_true_full"] for row in per_cell)
    gamma_e_true_robust = max(
        row["gamma_e_true_q95_q05"] for row in per_cell
    )
    gamma_e_hat = max(row["gamma_e_hat_full"] for row in per_cell)
    gamma_e_hat_robust = max(
        row["gamma_e_hat_q95_q05"] for row in per_cell
    )
    return {
        "available": True,
        "n_clusters_requested": int(n_clusters),
        "n_cells_used": int(len(per_cell)),
        "min_u_count": int(min_u_count),
        "diagnostic_propensity_clip": float(diagnostic_clip),
        "gamma_z_oracle_mc": float(gamma_z),
        "gamma_e_true_full": float(gamma_e_true),
        "gamma_e_true_q95_q05": float(gamma_e_true_robust),
        "gamma_e_hat_full": float(gamma_e_hat),
        "gamma_e_hat_q95_q05": float(gamma_e_hat_robust),
        "gamma_s_times_gamma_e_true": float(dgp.gamma_s * gamma_e_true),
        "gamma_s_times_gamma_e_hat": float(dgp.gamma_s * gamma_e_hat),
        "oracle_bound_holds_mc": bool(
            gamma_z <= dgp.gamma_s * gamma_e_true + 1e-8
        ),
        "cells": per_cell,
        "note": (
            "Quantizer fitted on training Z and evaluated on independent t=0 "
            "states, where the DGP's true e(S) formula is exact. The formal "
            "theorem remains a population statement; these are Monte Carlo/"
            "estimated analogues."
        ),
    }


def _lambda_seed(seed: int, lam: float) -> int:
    if lam == 0.0:
        return seed + 4000
    if lam == 5.0:
        return seed + 5000
    return seed + 6000 + int(lam * 1000)


def _checkpoint_payload(
    model: SPRL,
    *,
    seed: int,
    lam: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "seed": int(seed),
        "lambda_prop": float(lam),
        "latent_dim": int(args.latent_dim),
        "epochs": int(args.epochs),
        "n_trajectories": int(args.n_trajectories),
        "w_u": float(args.w_u),
        "conf_strength": float(args.conf_strength),
    }


def evaluate_representation(
    *,
    method: str,
    z_fit: np.ndarray,
    z_eval: np.ndarray,
    z_partition_eval: np.ndarray,
    model: SPRL | None,
    s_eval: np.ndarray,
    a_eval: np.ndarray,
    u_eval: np.ndarray,
    s_next_eval: np.ndarray,
    e_hat_eval: np.ndarray,
    s_partition_eval: np.ndarray,
    u_partition_eval: np.ndarray,
    e_hat_partition_eval: np.ndarray,
    e_true_partition_eval: np.ndarray,
    dgp: UnifiedCMDP,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    gamma_z_p95, gamma_info = compute_gamma_z_logistic(
        z_eval,
        a_eval,
        u_eval,
        percentile=95,
        use_interactions=True,
    )
    transition_mse = None
    if model is not None:
        transition_mse = compute_transition_mse(
            model, s_eval, a_eval, s_next_eval, device=args.device
        )

    return {
        "method": method,
        "gamma_s": float(dgp.gamma_s),
        "gamma_z_logistic_p95": float(gamma_z_p95),
        "gamma_z_logistic_median": float(gamma_info["gamma_z_median"]),
        "amplification_logistic_p95": float(gamma_z_p95 / dgp.gamma_s),
        "amplification_logistic_median": float(
            gamma_info["gamma_z_median"] / dgp.gamma_s
        ),
        "transition_mse": (
            None if transition_mse is None else float(transition_mse)
        ),
        "local_gamma_e_hat": local_gamma_e_diagnostic(
            z_eval,
            e_hat_eval,
            k_values=tuple(args.k_values),
            clip=args.propensity_clip,
            max_points=args.max_diagnostic_points,
            seed=seed,
        ),
        "partition": discrete_partition_diagnostic(
            z_fit,
            z_partition_eval,
            s_partition_eval,
            u_partition_eval,
            e_hat_partition_eval,
            e_true_partition_eval,
            dgp,
            n_clusters=args.n_clusters,
            min_u_count=args.min_u_count,
            diagnostic_clip=args.propensity_clip,
            seed=seed,
        ),
    }


def run_seed(
    seed: int,
    args: argparse.Namespace,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    start = time.perf_counter()
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
    train_trajectories = dgp.sample_trajectories(args.n_trajectories, rng=rng)
    train_flat = flatten_trajectories(train_trajectories)
    eval_rng = np.random.RandomState(args.eval_seed_offset + seed)
    eval_trajectories = dgp.sample_trajectories(
        args.eval_trajectories, rng=eval_rng
    )
    eval_flat = flatten_trajectories(eval_trajectories)

    s_train = train_flat["s"]
    a_train = train_flat["a"]
    r_train = train_flat["r"]
    s_next_train = train_flat["s_next"]

    s_eval = eval_flat["s"]
    a_eval = eval_flat["a"]
    u_eval = eval_flat["u"]
    s_next_eval = eval_flat["s_next"]
    # At t=0 the UnifiedCMDP posterior P(U|S) used by pi_b_marg is exact.
    # Later confounded dynamics invalidate that simple closed-form posterior,
    # so the theorem/true-e partition check is deliberately restricted to t=0.
    s_partition_eval = eval_trajectories["s"][:, 0, :]
    u_partition_eval = eval_trajectories["u"]

    torch.manual_seed(seed + 9000)
    prop_model = PropensityModel(dgp.state_dim).to(args.device)
    train_propensity(
        prop_model,
        s_train,
        a_train,
        epochs=args.prop_epochs,
        device=args.device,
        verbose=False,
    )
    e_hat_train = get_propensities(prop_model, s_train, device=args.device)
    e_hat_eval = get_propensities(prop_model, s_eval, device=args.device)
    e_hat_partition_eval = get_propensities(
        prop_model, s_partition_eval, device=args.device
    )
    e_true_partition_eval = dgp.pi_b_marg(s_partition_eval)
    torch.save(
        {
            "model_state_dict": prop_model.state_dict(),
            "seed": int(seed),
            "state_dim": int(dgp.state_dim),
            "prop_epochs": int(args.prop_epochs),
        },
        checkpoint_dir / f"seed{seed:03d}_propensity.pt",
    )

    result: dict[str, Any] = {
        "seed": int(seed),
        "gamma_s": float(dgp.gamma_s),
        "n_train_rows": int(len(s_train)),
        "n_eval_rows": int(len(s_eval)),
        "propensity_t0_eval_mae": float(
            np.mean(np.abs(e_hat_partition_eval - e_true_partition_eval))
        ),
        "methods": {},
    }

    # Raw representation is useful as a sanity check for the diagnostic, not
    # as a claimed compression certificate.
    result["methods"]["raw"] = evaluate_representation(
        method="raw",
        z_fit=s_train,
        z_eval=s_eval,
        z_partition_eval=s_partition_eval,
        model=None,
        s_eval=s_eval,
        a_eval=a_eval,
        u_eval=u_eval,
        s_next_eval=s_next_eval,
        e_hat_eval=e_hat_eval,
        s_partition_eval=s_partition_eval,
        u_partition_eval=u_partition_eval,
        e_hat_partition_eval=e_hat_partition_eval,
        e_true_partition_eval=e_true_partition_eval,
        dgp=dgp,
        seed=seed,
        args=args,
    )

    for lam in args.lambdas:
        torch.manual_seed(_lambda_seed(seed, lam))
        model = SPRL(dgp.state_dim, args.latent_dim, use_transitions=True).to(
            args.device
        )
        train_sprl(
            model,
            s_train,
            a_train,
            r_train,
            e_hat=e_hat_train if lam > 0 else np.zeros(len(s_train)),
            data_s_next=s_next_train,
            lambda_prop=lam,
            sigma=None,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs if lam > 0 else 0,
            lambda_ramp_epochs=args.lambda_ramp_epochs if lam > 0 else 0,
            grad_clip=1.0,
            device=args.device,
            verbose=False,
        )
        phi = get_sprl_encoder_fn(model, args.device)
        z_fit = phi(s_train)
        z_eval = phi(s_eval)
        z_partition_eval = phi(s_partition_eval)
        method = "ldm" if lam == 0.0 else f"sprl_lambda_{lam:g}"
        result["methods"][method] = evaluate_representation(
            method=method,
            z_fit=z_fit,
            z_eval=z_eval,
            z_partition_eval=z_partition_eval,
            model=model,
            s_eval=s_eval,
            a_eval=a_eval,
            u_eval=u_eval,
            s_next_eval=s_next_eval,
            e_hat_eval=e_hat_eval,
            s_partition_eval=s_partition_eval,
            u_partition_eval=u_partition_eval,
            e_hat_partition_eval=e_hat_partition_eval,
            e_true_partition_eval=e_true_partition_eval,
            dgp=dgp,
            seed=seed,
            args=args,
        )
        torch.save(
            _checkpoint_payload(model, seed=seed, lam=lam, args=args),
            checkpoint_dir / f"seed{seed:03d}_lambda{lam:g}.pt",
        )

    result["elapsed_seconds"] = float(time.perf_counter() - start)
    return result


def aggregate(seed_results: list[dict[str, Any]], k_for_summary: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed_result in seed_results:
        for method, metrics in seed_result["methods"].items():
            rows.append(
                {
                    "seed": seed_result["seed"],
                    "method": method,
                    "amplification_p95": metrics["amplification_logistic_p95"],
                    "amplification_median": metrics[
                        "amplification_logistic_median"
                    ],
                    "local_gamma_e_hat_p95": metrics["local_gamma_e_hat"][
                        str(k_for_summary)
                    ]["q95_q05_range"]["p95"],
                    "local_gamma_e_hat_full_p95": metrics["local_gamma_e_hat"][
                        str(k_for_summary)
                    ]["full_range"]["p95"],
                    "partition_gamma_e_hat": metrics["partition"][
                        "gamma_e_hat_full"
                    ],
                    "partition_gamma_e_hat_q95_q05": metrics["partition"][
                        "gamma_e_hat_q95_q05"
                    ],
                    "partition_gamma_e_true": metrics["partition"][
                        "gamma_e_true_full"
                    ],
                    "partition_gamma_e_true_q95_q05": metrics["partition"][
                        "gamma_e_true_q95_q05"
                    ],
                    "partition_gamma_z": metrics["partition"][
                        "gamma_z_oracle_mc"
                    ],
                    "partition_oracle_bound_holds": metrics["partition"][
                        "oracle_bound_holds_mc"
                    ],
                    "transition_mse": metrics["transition_mse"],
                }
            )

    methods = sorted({row["method"] for row in rows})
    by_method: dict[str, Any] = {}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        by_method[method] = {}
        for metric in (
            "amplification_p95",
            "amplification_median",
            "local_gamma_e_hat_p95",
            "local_gamma_e_hat_full_p95",
            "partition_gamma_e_hat",
            "partition_gamma_e_hat_q95_q05",
            "partition_gamma_e_true",
            "partition_gamma_e_true_q95_q05",
            "partition_gamma_z",
        ):
            available_values = [
                row[metric] for row in method_rows if row[metric] is not None
            ]
            by_method[method][metric] = (
                None
                if not available_values
                else summarize(np.asarray(available_values))
            )
        available_bounds = [
            row["partition_oracle_bound_holds"]
            for row in method_rows
            if row["partition_oracle_bound_holds"] is not None
        ]
        by_method[method]["partition_bound_success_rate"] = (
            None
            if not available_bounds
            else float(np.mean(available_bounds))
        )
        transition = [
            row["transition_mse"]
            for row in method_rows
            if row["transition_mse"] is not None
        ]
        by_method[method]["transition_mse"] = (
            None if not transition else summarize(np.asarray(transition))
        )

    def _spearman(selected: list[dict[str, Any]]) -> dict[str, float | None]:
        if len(selected) < 3:
            return {"rho": None, "p_value": None}
        selected_amp = np.asarray(
            [row["amplification_p95"] for row in selected]
        )
        selected_diagnostic = np.asarray(
            [row["local_gamma_e_hat_p95"] for row in selected]
        )
        rho_value, p_value_value = spearmanr(
            selected_diagnostic, selected_amp
        )
        if not np.isfinite(rho_value):
            return {"rho": None, "p_value": None}
        return {
            "rho": float(rho_value),
            "p_value": float(p_value_value),
        }

    learned_rows = [row for row in rows if row["method"] != "raw"]
    per_seed_rank: list[dict[str, Any]] = []
    for seed in sorted({row["seed"] for row in learned_rows}):
        selected = [row for row in learned_rows if row["seed"] == seed]
        rank = _spearman(selected)
        best_diagnostic = min(
            selected, key=lambda row: row["local_gamma_e_hat_p95"]
        )["method"]
        best_oracle = min(
            selected, key=lambda row: row["amplification_p95"]
        )["method"]
        per_seed_rank.append(
            {
                "seed": int(seed),
                **rank,
                "best_observable_diagnostic": best_diagnostic,
                "best_oracle_amplification": best_oracle,
                "best_match": bool(best_diagnostic == best_oracle),
            }
        )

    finite_seed_rhos = [
        row["rho"] for row in per_seed_rank if row["rho"] is not None
    ]
    return {
        "n_rows": int(len(rows)),
        "k_for_summary": int(k_for_summary),
        "by_method": by_method,
        "pooled_spearman_all_rows": _spearman(rows),
        "pooled_spearman_learned_rows": _spearman(learned_rows),
        "within_seed_lambda_rank": per_seed_rank,
        "within_seed_spearman_mean": (
            None if not finite_seed_rhos else float(np.mean(finite_seed_rhos))
        ),
        "best_lambda_match_rate": (
            None
            if not per_seed_rank
            else float(np.mean([row["best_match"] for row in per_seed_rank]))
        ),
        "partition_bound_success_rate": (
            None
            if not [
                row["partition_oracle_bound_holds"]
                for row in rows
                if row["partition_oracle_bound_holds"] is not None
            ]
            else float(
                np.mean(
                    [
                        row["partition_oracle_bound_holds"]
                        for row in rows
                        if row["partition_oracle_bound_holds"] is not None
                    ]
                )
            )
        ),
        "rows": rows,
    }


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--seed_start", type=int, default=1)
    parser.add_argument("--w_u", type=float, default=0.5)
    parser.add_argument("--conf_strength", type=float, default=2.0)
    parser.add_argument("--n_trajectories", type=int, default=5000)
    parser.add_argument("--eval_trajectories", type=int, default=4000)
    parser.add_argument("--eval_seed_offset", type=int, default=100000)
    parser.add_argument("--latent_dim", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--prop_epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=50)
    parser.add_argument("--lambda_ramp_epochs", type=int, default=50)
    parser.add_argument(
        "--lambdas", type=float, nargs="+", default=list(DEFAULT_LAMBDAS)
    )
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--min_u_count", type=int, default=50)
    parser.add_argument("--propensity_clip", type=float, default=0.02)
    parser.add_argument(
        "--k_values", type=int, nargs="+", default=[25, 50, 100]
    )
    parser.add_argument("--max_diagnostic_points", type=int, default=10000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing output file and skip completed seed IDs.",
    )
    parser.add_argument(
        "--output_dir", default="outputs/propensity_diagnostic"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    args.lambdas = tuple(float(value) for value in args.lambdas)

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gamma_e_diagnostic.json"

    if args.resume and output_path.exists():
        with output_path.open() as handle:
            payload = json.load(handle)
        completed_seed_ids = {
            int(result["seed"]) for result in payload["seed_results"]
        }
        print(
            f"Resuming {output_path}; completed seeds="
            f"{sorted(completed_seed_ids)}",
            flush=True,
        )
    else:
        payload = {
            "config": {
                key: value
                for key, value in vars(args).items()
                if key not in ("output_dir", "resume")
            },
            "interpretation": {
                "discrete_partition": (
                    "Monte Carlo validation of the binary-U theorem after "
                    "explicit k-means quantization."
                ),
                "continuous_local": (
                    "Observable kNN risk diagnostic only; not a formal bound "
                    "for the original continuous representation."
                ),
            },
            "seed_results": [],
            "aggregate": None,
        }
        completed_seed_ids = set()

    for offset in range(args.K):
        seed = args.seed_start + offset
        if seed in completed_seed_ids:
            print(f"Skipping completed seed {seed}", flush=True)
            continue
        print(f"Running diagnostic seed {seed} ({offset + 1}/{args.K})", flush=True)
        seed_result = run_seed(seed, args, checkpoint_dir)
        payload["seed_results"].append(seed_result)
        payload["aggregate"] = aggregate(
            payload["seed_results"], k_for_summary=max(args.k_values)
        )
        atomic_json_dump(payload, output_path)
        print(
            f"  finished in {seed_result['elapsed_seconds'] / 60:.1f} min; "
            f"saved {output_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
