"""Exact H=2 outcome-confounded sensitivity-envelope construction.

This is deliberately a tiny enumerable construction.  It reuses the submitted
Appendix-D state/action mechanism and compression, but makes the following
sequential extension:

* U is sampled once.
* The base component of S_0 has the submitted joint distribution with U.
  The next state appends the first action, S_1=(S_0,A_0), so the transition
  kernel is deterministic and has no direct U dependence given (S_0,A_0).
* A_0 and A_1 are drawn independently conditional on (S_0, U) from the
  submitted behavior policy.
* The terminal potential reward is
  Y(a_0,a_1) = 1{a_0=a_1=1}(1-U).  Hence both target actions matter and
  there is genuine hidden outcome confounding, while the reward law is
  invariant across raw states conditional on (Z,U).
* The target policy deterministically chooses action 1 twice.  It is both
  S-measurable and Z-measurable, and its exact value is 1/2.

The interval below is a valid but generally conservative sequential
Horvitz-Thompson envelope, not a sharp sequential MSM bound.  At every observed
history h_t it assumes the standard marginal sensitivity model

    1/Gamma <= odds(p_t(1 | h_t,U)) / odds(e_t(1 | h_t)) <= Gamma,

where e_t is the observed history-conditional propensity.  On target-matching
trajectories the unknown likelihood-ratio factor satisfies

    e_t + (1-e_t)/Gamma
      <= e_t / p_t(1 | h_t,U)
      <= e_t + Gamma(1-e_t).

For nonnegative Y, multiplying these pointwise bounds over t gives an outer
interval for the exact sequential IPW identity.  No one-step interval is
silently reused.

The submission states the MSM as a pairwise odds bound across u,u'.  That
pairwise bound implies the same-Gamma deviation bound from the observed
history-conditional propensity because the latter is a mixture of the
u-specific probabilities and its odds lie between their odds.  Therefore the
relative-to-marginal boxes used here are a conservative relaxation of the
submission's history-conditional pairwise model.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dgp.discrete_counterexample import build_counterexample


def odds(probability: float) -> float:
    probability = float(np.clip(probability, 1e-15, 1.0 - 1e-15))
    return probability / (1.0 - probability)


def action_probability(probability_one: float, action: int) -> float:
    return probability_one if action == 1 else 1.0 - probability_one


def state_cell(dgp, state: int, space: str) -> int:
    if space == "raw":
        return int(state)
    if space == "latent":
        return int(dgp.latent_indices.index(dgp.compression_map[state]))
    raise ValueError(space)


def history_mass(
    dgp,
    space: str,
    cell: int,
    prefix: tuple[int, ...],
    u_value: int | None,
) -> float:
    """P(X_0=cell,A_0:t-1=prefix[,U=u]) under the behavior law."""
    total = 0.0
    for state in range(dgp.n_states):
        if state_cell(dgp, state, space) != cell:
            continue
        for u in range(dgp.n_u):
            if u_value is not None and u != u_value:
                continue
            mass = float(dgp.p_su[state, u])
            for action in prefix:
                mass *= action_probability(float(dgp.pi_b[state, u]), action)
            total += mass
    return total


def history_propensity(
    dgp,
    space: str,
    cell: int,
    prefix: tuple[int, ...],
    u_value: int | None,
) -> float:
    """P(A_t=1 | X_0=cell,A_0:t-1=prefix[,U=u])."""
    denominator = history_mass(dgp, space, cell, prefix, u_value)
    if denominator <= 0.0:
        raise ValueError("Zero-support history")
    numerator = 0.0
    for state in range(dgp.n_states):
        if state_cell(dgp, state, space) != cell:
            continue
        for u in range(dgp.n_u):
            if u_value is not None and u != u_value:
                continue
            mass = float(dgp.p_su[state, u])
            for action in prefix:
                mass *= action_probability(float(dgp.pi_b[state, u]), action)
            numerator += mass * float(dgp.pi_b[state, u])
    return numerator / denominator


def cells(dgp, space: str) -> Iterable[int]:
    return range(dgp.n_states if space == "raw" else dgp.n_latent)


def sensitivity_levels(dgp, space: str, horizon: int) -> dict:
    """Return nominal-to-marginal and pairwise history-conditional levels."""
    nominal = 1.0
    pairwise = 1.0
    maximizer_nominal: dict | None = None
    maximizer_pairwise: dict | None = None
    records = []

    for t in range(horizon):
        for prefix in product((0, 1), repeat=t):
            for cell in cells(dgp, space):
                marginal = history_propensity(dgp, space, cell, prefix, None)
                conditional = [
                    history_propensity(dgp, space, cell, prefix, u)
                    for u in range(dgp.n_u)
                ]
                for u, probability in enumerate(conditional):
                    ratio = odds(probability) / odds(marginal)
                    deviation = max(ratio, 1.0 / ratio)
                    if deviation > nominal:
                        nominal = deviation
                        maximizer_nominal = {
                            "t": t,
                            "prefix": list(prefix),
                            "cell": cell,
                            "u": u,
                            "e_marginal": marginal,
                            "p_conditional": probability,
                        }
                for u in range(dgp.n_u):
                    for u_prime in range(dgp.n_u):
                        ratio = odds(conditional[u]) / odds(conditional[u_prime])
                        if ratio > pairwise:
                            pairwise = ratio
                            maximizer_pairwise = {
                                "t": t,
                                "prefix": list(prefix),
                                "cell": cell,
                                "u": u,
                                "u_prime": u_prime,
                                "p_u": conditional[u],
                                "p_u_prime": conditional[u_prime],
                            }
                records.append(
                    {
                        "t": t,
                        "prefix": list(prefix),
                        "cell": cell,
                        "e_marginal": marginal,
                        "p_conditional_by_u": conditional,
                    }
                )

    return {
        "nominal_to_marginal": nominal,
        "pairwise": pairwise,
        "nominal_maximizer": maximizer_nominal,
        "pairwise_maximizer": maximizer_pairwise,
        "histories": records,
    }


def observed_target_mass(dgp, space: str, cell: int, u: int, horizon: int) -> float:
    """P(X_0=cell,U=u,A_0:H-1=1) under the behavior law."""
    total = 0.0
    for state in range(dgp.n_states):
        if state_cell(dgp, state, space) != cell:
            continue
        total += float(dgp.p_su[state, u]) * float(dgp.pi_b[state, u]) ** horizon
    return total


def target_sequence_propensity(
    dgp,
    space: str,
    cell: int,
    u_value: int | None,
    horizon: int,
) -> float:
    probability = 1.0
    for t in range(horizon):
        prefix = (1,) * t
        probability *= history_propensity(
            dgp, space, cell, prefix, u_value
        )
    return probability


def exact_sequential_ipw_value(dgp, space: str, horizon: int) -> float:
    """Evaluate the exact latent-U sequential IPW identity."""
    value = 0.0
    for cell in cells(dgp, space):
        for u in range(dgp.n_u):
            observed_mass = observed_target_mass(dgp, space, cell, u, horizon)
            sequence_probability = target_sequence_propensity(
                dgp, space, cell, u, horizon
            )
            reward = float(1 - u)
            value += observed_mass * reward / sequence_probability
    return value


def sequential_ht_envelope(
    dgp,
    space: str,
    horizon: int,
    gamma: float,
) -> tuple[float, float, dict]:
    """Conservative sequential MSM envelope for the all-one target."""
    lower = 0.0
    upper = 0.0
    details = []

    for cell in cells(dgp, space):
        observed_sequence_probability = target_sequence_propensity(
            dgp, space, cell, None, horizon
        )
        lower_ratio = 1.0
        upper_ratio = 1.0
        propensities = []
        for t in range(horizon):
            prefix = (1,) * t
            e_t = history_propensity(dgp, space, cell, prefix, None)
            lower_ratio *= e_t + (1.0 - e_t) / gamma
            upper_ratio *= e_t + gamma * (1.0 - e_t)
            propensities.append(e_t)

        observed_reward_mass = sum(
            observed_target_mass(dgp, space, cell, u, horizon) * float(1 - u)
            for u in range(dgp.n_u)
        )
        base = observed_reward_mass / observed_sequence_probability
        cell_lower = base * lower_ratio
        cell_upper = base * upper_ratio
        lower += cell_lower
        upper += cell_upper
        details.append(
            {
                "cell": cell,
                "target_path_marginal_propensities": propensities,
                "observed_target_sequence_probability": observed_sequence_probability,
                "observed_reward_mass": observed_reward_mass,
                "base_ht_contribution": base,
                "likelihood_ratio_lower": lower_ratio,
                "likelihood_ratio_upper": upper_ratio,
                "lower_contribution": cell_lower,
                "upper_contribution": cell_upper,
            }
        )

    # The terminal reward is known to lie in [0,1].
    unclipped = (lower, upper)
    lower = max(0.0, lower)
    upper = min(1.0, upper)
    return lower, upper, {
        "unclipped": list(unclipped),
        "support_clipped": [lower, upper],
        "cells": details,
    }


def run(horizon: int = 2) -> dict:
    if horizon != 2:
        raise ValueError("This audited construction is defined for exactly H=2.")

    dgp = build_counterexample(
        gamma_s_target=2.0,
        alpha=0.95,
        q=0.9,
        r=0.1,
    )
    true_value = float(dgp.p_u[0])
    submitted_gamma_s = float(dgp.compute_gamma_s())

    levels = {
        space: sensitivity_levels(dgp, space, horizon)
        for space in ("raw", "latent")
    }
    exact_ipw = {
        space: exact_sequential_ipw_value(dgp, space, horizon)
        for space in ("raw", "latent")
    }

    nominal_bounds = {}
    corrected_bounds = {}
    for space in ("raw", "latent"):
        lo, hi, info = sequential_ht_envelope(
            dgp, space, horizon, submitted_gamma_s
        )
        nominal_bounds[space] = {"lower": lo, "upper": hi, "info": info}
        corrected_gamma = levels[space]["nominal_to_marginal"]
        lo_c, hi_c, info_c = sequential_ht_envelope(
            dgp, space, horizon, corrected_gamma
        )
        corrected_bounds[space] = {
            "gamma": corrected_gamma,
            "lower": lo_c,
            "upper": hi_c,
            "info": info_c,
        }

    def covers(bounds: dict) -> bool:
        return bool(bounds["lower"] <= true_value <= bounds["upper"])

    checks = {
        "raw_exact_ipw_matches_truth": bool(
            np.isclose(exact_ipw["raw"], true_value, atol=1e-12)
        ),
        "latent_exact_ipw_matches_truth": bool(
            np.isclose(exact_ipw["latent"], true_value, atol=1e-12)
        ),
        "raw_nominal_model_holds": bool(
            levels["raw"]["nominal_to_marginal"] <= submitted_gamma_s + 1e-12
        ),
        "latent_nominal_model_fails": bool(
            levels["latent"]["nominal_to_marginal"] > submitted_gamma_s + 1e-12
        ),
        "raw_nominal_bounds_cover": covers(nominal_bounds["raw"]),
        "latent_nominal_bounds_exclude_truth": not covers(
            nominal_bounds["latent"]
        ),
        "latent_corrected_bounds_cover": covers(corrected_bounds["latent"]),
        "raw_nominal_bounds_nonvacuous": bool(
            nominal_bounds["raw"]["upper"] - nominal_bounds["raw"]["lower"] < 1.0
        ),
        "latent_nominal_bounds_nonvacuous": bool(
            nominal_bounds["latent"]["upper"]
            - nominal_bounds["latent"]["lower"]
            < 1.0
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"An audited multi-step check failed: {checks}")

    return {
        "protocol": {
            "horizon": horizon,
            "source_constructor": "dgp.discrete_counterexample.build_counterexample",
            "source_parameters": {
                "gamma_s_target": 2.0,
                "alpha": 0.95,
                "q": 0.9,
                "r": 0.1,
            },
            "hidden_confounder": "U in {0,1}, fixed per trajectory",
            "transition": (
                "S_1=(S_0,A_0) (deterministic; no direct U effect)"
            ),
            "behavior": "Appendix-D pi_b(A_t=1|S_0,U), conditionally independent over t",
            "terminal_potential_reward": (
                "Y(a_0,a_1)=1{a_0=a_1=1}(1-U)"
            ),
            "target_policy": "A_t=1 at every t",
            "compression": "submitted phi(s1)=phi(s2)=z1, phi(s3)=z3",
            "bound": (
                "support-clipped conservative sequential HT outer envelope "
                "under a history-conditional marginal sensitivity model; not sharp"
            ),
        },
        "population_tables": {
            "p_su": dgp.p_su.tolist(),
            "pi_b_action_one_given_s_u": dgp.pi_b.tolist(),
            "compression_map": {
                str(state): int(latent)
                for state, latent in dgp.compression_map.items()
            },
        },
        "true_value": true_value,
        "submitted_pairwise_gamma_s": submitted_gamma_s,
        "sensitivity_levels": levels,
        "exact_sequential_ipw_value": exact_ipw,
        "bounds_at_submitted_gamma_s": nominal_bounds,
        "bounds_at_required_history_conditional_gamma": corrected_bounds,
        "checks": checks,
    }


def render_report(result: dict) -> str:
    gamma_s = result["submitted_pairwise_gamma_s"]
    true_value = result["true_value"]
    raw_level = result["sensitivity_levels"]["raw"]["nominal_to_marginal"]
    latent_level = result["sensitivity_levels"]["latent"]["nominal_to_marginal"]
    raw_pairwise = result["sensitivity_levels"]["raw"]["pairwise"]
    latent_pairwise = result["sensitivity_levels"]["latent"]["pairwise"]
    raw = result["bounds_at_submitted_gamma_s"]["raw"]
    latent = result["bounds_at_submitted_gamma_s"]["latent"]
    latent_corrected = result["bounds_at_required_history_conditional_gamma"][
        "latent"
    ]
    checks = result["checks"]

    return rf"""# Exact direct multi-step coverage audit

## Result

For the enumerable \(H={result['protocol']['horizon']}\) construction, the
always-action-1 target value is exactly {true_value:.6f}.  At the submitted
raw-state pairwise level \(\Gamma_S={gamma_s:.6f}\), the conservative
history-conditional sequential HT envelopes are:

| State used by the analyst | Sensitivity level | Interval | Covers? | Width |
|---|---:|---:|---:|---:|
| Raw \(S\) | {gamma_s:.6f} | [{raw['lower']:.6f}, {raw['upper']:.6f}] | {checks['raw_nominal_bounds_cover']} | {raw['upper']-raw['lower']:.6f} |
| Compressed \(Z\) | {gamma_s:.6f} | [{latent['lower']:.6f}, {latent['upper']:.6f}] | {not checks['latent_nominal_bounds_exclude_truth']} | {latent['upper']-latent['lower']:.6f} |
| Compressed \(Z\) | corrected {latent_corrected['gamma']:.6f} | [{latent_corrected['lower']:.6f}, {latent_corrected['upper']:.6f}] | {checks['latent_corrected_bounds_cover']} | {latent_corrected['upper']-latent_corrected['lower']:.6f} |

Thus the raw-state interval is nonvacuous and covers, whereas compression at
the same nominal level excludes the truth.  Raising the latent
history-conditional level to the level actually required by the constructed
DGP restores coverage (although conservatism can make the corrected interval
wide).

## Sensitivity transport

The maximum history-conditional odds deviations from the observed marginal
propensity are:

| Space | Nominal-to-marginal level | Pairwise \(U\)-level |
|---|---:|---:|
| Raw \(S\) | {raw_level:.6f} | {raw_pairwise:.6f} |
| Compressed \(Z\) | {latent_level:.6f} | {latent_pairwise:.6f} |

The submitted pairwise \(\Gamma_S=2\) safely upper-bounds every raw
history-conditional nominal deviation, but not the compressed deviations.
The exact latent-\(U\) sequential IPW identities evaluate to
{result['exact_sequential_ipw_value']['raw']:.6f} in raw space and
{result['exact_sequential_ipw_value']['latent']:.6f} in latent space, matching
the true value in both parameterizations.

## Construction and estimand

- \(U\) is drawn once per trajectory.
- The source parameters are exactly
  \(\Gamma_S=2,\alpha=0.95,q=0.9,r=0.1\); the JSON artifact records the full
  \(P(S,U)\), behavior-policy, and compression tables.
- The base state \(S_0\) follows the submitted Appendix-D joint law with
  \(U\), and \(S_1=(S_0,A_0)\).  Thus the transition is deterministic and has
  no direct \(U\) dependence given the current raw state and action.  The
  latent map retains the action component:
  \(Z_0=\phi(S_0), Z_1=(\phi(S_0),A_0)\).
- The two behavior actions are conditionally independent given \((S_0,U)\)
  and use the submitted behavior policy.
- The terminal potential reward is
  \(Y(a_0,a_1)=\mathbf{{1}}\{{a_0=a_1=1\}}(1-U)\).  Both target actions
  therefore matter.  This creates genuine hidden outcome confounding while
  making discarded raw state identity outcome-irrelevant conditional on
  \((Z,U)\).
- The target chooses action 1 twice, so it is measurable in both state spaces.

## What the interval is

This is a direct **sequential** calculation.  It uses the exact chain of
observed history-conditional propensities and multiplies the corresponding
per-step likelihood-ratio boxes along the target path.  For nonnegative
terminal reward, this gives a valid outer interval under the stated
history-conditional MSM.  The interval is clipped only by the known reward
support \([0,1]\).

The paper writes its MSM as a pairwise odds bound over \(u,u'\).  At each
history, the observed propensity is a mixture of the \(u\)-specific
propensities, so its odds lie between their odds.  Consequently, a pairwise
level \(\Gamma\) implies the same-\(\Gamma\) box relative to the observed
propensity used by this envelope.  This is why applying the submitted
\(\Gamma_S=2\) yields a valid (possibly conservative) raw-state interval.

It is **not** claimed to be sharp, and it is not Equation 11 or a one-step
bound applied to flattened transitions.  No result here covers transition
confounding \(U\to S_{{t+1}}\); that regime needs a separate transition
sensitivity model.

This is an explicit outcome-confounded extension, not an instance of the
submission's reward-confounding separation: \(Y(1,1)=1-U\). Its structural
outcome-sufficiency statement is
\(Y(a_0,a_1)\perp S_0\mid(Z_0,U)\), because discarded state identity never
enters the potential outcome. It does **not** claim the observational
relation \(R\perp S\mid(Z,A)\), which generally fails because \(S\) changes
the posterior distribution of \(U\).

Finally, the corrected level `{latent_level:.6f}` is the maximum
history-conditional odds deviation from the observed marginal propensity
used by this sequential envelope. The corresponding pairwise latent
\(U\)-level is `{latent_pairwise:.6f}`. These are related sensitivity
conventions, not two estimates of the same \(\Gamma_Z\), and must retain
their labels.

## Automated checks

```text
{json.dumps(checks, indent=2, sort_keys=True)}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/exact_outcome_h2"),
    )
    args = parser.parse_args()

    result = run(horizon=args.horizon)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "multistep_coverage_exact.json"
    report_path = args.output_dir / "multistep_coverage_exact.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report_path.write_text(render_report(result))

    print(render_report(result))
    print(f"\nJSON: {json_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
