"""Exact H=2 transition-confounded sensitivity-transport calculation.

This analysis uses a transition-kernel sensitivity model, denoted Lambda_T,
that is deliberately distinct from the paper's action-odds Gamma:

    1 / Lambda_T <= T^do(y | x, a) / T^obs(y | x, a) <= Lambda_T.

Here T^do is the interventional transition row relevant to the target action
and T^obs is the observed behavior-conditioned row.  For a binary next-state
probability q=T^obs(Y=1|x,a), the sharp interval under this direct rectangular
transition model is

    max(q/Lambda_T, 1-Lambda_T(1-q))
      <= T^do(Y=1|x,a)
      <= min(Lambda_T q, 1-(1-q)/Lambda_T).

No action-sensitivity and transition-sensitivity parameters are multiplied:
doing so would require a separately specified joint sensitivity model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dgp.transition_confounded_h2 import TransitionConfoundedH2DGP


def odds(probability: Fraction) -> Fraction:
    if probability <= 0 or probability >= 1:
        raise ValueError("Odds require a probability strictly between zero and one")
    return probability / (1 - probability)


def symmetric_ratio(left: Fraction, right: Fraction) -> Fraction:
    if left <= 0 or right <= 0:
        raise ValueError("Density ratios require positive entries")
    return max(left / right, right / left)


def bernoulli_row_level(
    interventional_y1: Fraction, observed_y1: Fraction
) -> Fraction:
    """Smallest multiplicative level covering both binary outcomes."""
    return max(
        symmetric_ratio(interventional_y1, observed_y1),
        symmetric_ratio(1 - interventional_y1, 1 - observed_y1),
    )


def binary_row_bounds(
    observed_y1: Fraction, level: Fraction
) -> tuple[Fraction, Fraction]:
    """Sharp row-rectangular bounds for a binary transition probability."""
    if level < 1:
        raise ValueError("Sensitivity level must be at least one")
    lower = max(
        observed_y1 / level,
        1 - level * (1 - observed_y1),
        Fraction(0),
    )
    upper = min(
        level * observed_y1,
        1 - (1 - observed_y1) / level,
        Fraction(1),
    )
    if lower > upper:
        raise AssertionError("Sensitivity row is empty")
    return lower, upper


def pairwise_action_level(probabilities: Iterable[Fraction]) -> Fraction:
    probability_list = list(probabilities)
    return max(
        symmetric_ratio(odds(left), odds(right))
        for left in probability_list
        for right in probability_list
    )


def weighted_interval(
    weights: Iterable[Fraction],
    intervals: Iterable[tuple[Fraction, Fraction]],
) -> tuple[Fraction, Fraction]:
    pairs = list(zip(weights, intervals, strict=True))
    return (
        sum(weight * interval[0] for weight, interval in pairs),
        sum(weight * interval[1] for weight, interval in pairs),
    )


def fraction_record(value: Fraction) -> dict:
    return {
        "fraction": f"{value.numerator}/{value.denominator}",
        "value": float(value),
    }


def interval_record(interval: tuple[Fraction, Fraction]) -> dict:
    lower, upper = interval
    return {
        "lower": fraction_record(lower),
        "upper": fraction_record(upper),
        "width": fraction_record(upper - lower),
    }


def table_record(values: Iterable[Fraction]) -> list[dict]:
    return [fraction_record(value) for value in values]


def covers(interval: tuple[Fraction, Fraction], value: Fraction) -> bool:
    return interval[0] <= value <= interval[1]


def analyze(
    dgp: TransitionConfoundedH2DGP,
    *,
    enforce_stress_pattern: bool = True,
) -> dict:
    true_value = dgp.target_value_q()

    causal_raw = [
        dgp.causal_transition_y1_given_state_q(state, action=1)
        for state in range(dgp.n_states)
    ]
    observed_raw = [
        dgp.observed_transition_y1_given_state_q(state, action=1)
        for state in range(dgp.n_states)
    ]
    causal_latent = dgp.causal_transition_y1_latent_q(action=1)
    observed_latent = dgp.observed_transition_y1_latent_q(action=1)

    raw_transition_level = max(
        bernoulli_row_level(causal, observed)
        for causal, observed in zip(causal_raw, observed_raw, strict=True)
    )
    latent_transition_level = bernoulli_row_level(
        causal_latent, observed_latent
    )
    raw_hidden_u_transition_level = max(
        bernoulli_row_level(
            dgp.transition_y1_q[1][state][0],
            dgp.transition_y1_q[1][state][1],
        )
        for state in range(dgp.n_states)
    )
    latent_transition_given_u = [
        sum(
            dgp.p_su_q[state][u] * dgp.transition_y1_q[1][state][u]
            for state in range(dgp.n_states)
        )
        / dgp.hidden_mass_q(u)
        for u in range(dgp.n_u)
    ]
    latent_hidden_u_transition_level = bernoulli_row_level(
        latent_transition_given_u[0],
        latent_transition_given_u[1],
    )

    state_weights = [
        dgp.state_mass_q(state) for state in range(dgp.n_states)
    ]
    raw_row_intervals = [
        binary_row_bounds(observed, raw_transition_level)
        for observed in observed_raw
    ]
    raw_interval = weighted_interval(state_weights, raw_row_intervals)
    latent_at_raw_interval = binary_row_bounds(
        observed_latent, raw_transition_level
    )
    latent_corrected_interval = binary_row_bounds(
        observed_latent, latent_transition_level
    )

    raw_action_level = max(
        pairwise_action_level(dgp.pi0_q[state])
        for state in range(dgp.n_states)
    )
    latent_action_probabilities = [
        dgp.latent_action_one_given_u_q(u) for u in range(dgp.n_u)
    ]
    latent_action_level = pairwise_action_level(latent_action_probabilities)

    behavior_action_mass_by_state = [
        sum(
            dgp.p_su_q[state][u] * dgp.pi0_q[state][u]
            for u in range(dgp.n_u)
        )
        for state in range(dgp.n_states)
    ]
    behavior_action_mass = sum(behavior_action_mass_by_state)
    behavior_selected_weights = [
        mass / behavior_action_mass for mass in behavior_action_mass_by_state
    ]
    common_weight_causal_y1 = sum(
        weight * causal
        for weight, causal in zip(
            behavior_selected_weights, causal_raw, strict=True
        )
    )
    composition_level = bernoulli_row_level(
        common_weight_causal_y1, causal_latent
    )
    conservative_product_level = raw_transition_level * composition_level
    conservative_product_interval = binary_row_bounds(
        observed_latent, conservative_product_level
    )

    row_factor_y1 = observed_latent / common_weight_causal_y1
    composition_factor_y1 = common_weight_causal_y1 / causal_latent
    path_ratio_example = (
        dgp.p_su_q[1][1]
        / dgp.p_su_q[1][0]
        * (dgp.pi0_q[1][1] / dgp.pi0_q[1][0])
        * (
            dgp.transition_y1_q[1][1][1]
            / dgp.transition_y1_q[1][1][0]
        )
        * (dgp.pi1_q / dgp.pi1_q)
    )

    oracle_ipw = sum(
        record["probability"]
        * record["reward"]
        / dgp.pi0_q[record["state0"]][record["u"]]
        / dgp.pi1_q
        for record in dgp.enumerate_behavior_trajectories()
        if record["action0"] == 1 and record["action1"] == 1
    )

    checks = {
        "behavior_trajectory_mass_is_one": (
            sum(
                record["probability"]
                for record in dgp.enumerate_behavior_trajectories()
            )
            == 1
        ),
        "genuine_raw_transition_confounding": (
            dgp.transition_y1_q[1][1][0]
            != dgp.transition_y1_q[1][1][1]
        ),
        "first_action_affects_transition": any(
            dgp.transition_y1_q[0][state][u]
            != dgp.transition_y1_q[1][state][u]
            for state in range(dgp.n_states)
            for u in range(dgp.n_u)
        ),
        "second_action_affects_reward": any(
            dgp.reward(state, u, 0, y) != dgp.reward(state, u, 1, y)
            for state in range(dgp.n_states)
            for u in range(dgp.n_u)
            for y in range(2)
        ),
        "reward_has_no_direct_raw_state_dependence": all(
            dgp.reward(0, u, action1, y)
            == dgp.reward(1, u, action1, y)
            for u in range(dgp.n_u)
            for action1 in range(2)
            for y in range(2)
        ),
        "target_value_matches_oracle_ipw": oracle_ipw == true_value,
        "raw_transition_level_is_121_over_105": (
            raw_transition_level == Fraction(121, 105)
        ),
        "latent_transition_level_is_205_over_117": (
            latent_transition_level == Fraction(205, 117)
        ),
        "transition_sensitivity_amplifies": (
            latent_transition_level > raw_transition_level
        ),
        "action_sensitivity_does_not_amplify": (
            latent_action_level < raw_action_level
        ),
        "pairwise_hidden_u_transition_level_does_not_amplify": (
            latent_hidden_u_transition_level < raw_hidden_u_transition_level
        ),
        "raw_interval_covers": covers(raw_interval, true_value),
        "compressed_interval_at_raw_level_excludes": not covers(
            latent_at_raw_interval, true_value
        ),
        "compressed_interval_excludes_from_below": (
            latent_at_raw_interval[0] > true_value
        ),
        "corrected_compressed_interval_covers": covers(
            latent_corrected_interval, true_value
        ),
        "raw_interval_nonvacuous": raw_interval[1] - raw_interval[0] < 1,
        "compressed_interval_nonvacuous": (
            latent_at_raw_interval[1] - latent_at_raw_interval[0] < 1
        ),
        "tight_y1_factorization": (
            row_factor_y1 * composition_factor_y1
            == observed_latent / causal_latent
        ),
        "product_bound_dominates_latent_level": (
            conservative_product_level >= latent_transition_level
        ),
        "product_bound_interval_covers": covers(
            conservative_product_interval, true_value
        ),
        "path_ratio_example_is_24": path_ratio_example == 24,
    }
    universal_checks = (
        "behavior_trajectory_mass_is_one",
        "first_action_affects_transition",
        "second_action_affects_reward",
        "reward_has_no_direct_raw_state_dependence",
        "target_value_matches_oracle_ipw",
        "raw_interval_covers",
        "corrected_compressed_interval_covers",
        "raw_interval_nonvacuous",
        "compressed_interval_nonvacuous",
        "tight_y1_factorization",
        "product_bound_dominates_latent_level",
        "product_bound_interval_covers",
    )
    required_checks = (
        tuple(checks) if enforce_stress_pattern else universal_checks
    )
    failed = [name for name in required_checks if not checks[name]]
    if failed:
        raise AssertionError(f"Transition-confounded H=2 checks failed: {failed}")

    return {
        "protocol": {
            "analysis_version": 1,
            "horizon": 2,
            "hidden_confounder": "binary U fixed per trajectory",
            "raw_states": "S0=s and S1=(s,Y)",
            "latent_states": "Z0 is one merged cell and Z1=Y",
            "behavior": (
                "A0 follows pi0(S0,U); A1 is Bernoulli(1/2) independently"
            ),
            "transition": (
                "Y is drawn from T(Y|S0,A0,U), with direct U dependence "
                "at (S0=1,A0=1)"
            ),
            "reward": "R=A1*Y",
            "target_policy": "A0=A1=1",
            "transition_sensitivity_model": (
                "multiplicative interventional-to-observed transition-row "
                "density ratio Lambda_T"
            ),
            "separation_from_action_model": (
                "Lambda_T is not the paper's action-odds Gamma; the two "
                "parameters are not multiplied without a joint model"
            ),
        },
        "population_tables": {
            "p_su": [
                table_record(row) for row in dgp.p_su_q
            ],
            "pi0_action_one_given_s_u": [
                table_record(row) for row in dgp.pi0_q
            ],
            "transition_y1_given_a_s_u": [
                [table_record(row) for row in matrix]
                for matrix in dgp.transition_y1_q
            ],
            "pi1_action_one": fraction_record(dgp.pi1_q),
        },
        "target_value": fraction_record(true_value),
        "transition_rows": {
            "causal_raw_y1": table_record(causal_raw),
            "observed_raw_y1": table_record(observed_raw),
            "causal_latent_y1": fraction_record(causal_latent),
            "observed_latent_y1": fraction_record(observed_latent),
        },
        "sensitivity_levels": {
            "transition_raw_lambda_t": fraction_record(raw_transition_level),
            "transition_latent_lambda_t": fraction_record(
                latent_transition_level
            ),
            "action_raw_pairwise_gamma": fraction_record(raw_action_level),
            "action_latent_pairwise_gamma": fraction_record(
                latent_action_level
            ),
            "transition_raw_pairwise_hidden_u": fraction_record(
                raw_hidden_u_transition_level
            ),
            "transition_latent_pairwise_hidden_u": fraction_record(
                latent_hidden_u_transition_level
            ),
        },
        "robust_value_intervals": {
            "raw_at_raw_lambda_t": interval_record(raw_interval),
            "compressed_at_raw_lambda_t": interval_record(
                latent_at_raw_interval
            ),
            "compressed_at_required_lambda_t": interval_record(
                latent_corrected_interval
            ),
            "compressed_at_conservative_product_level": interval_record(
                conservative_product_interval
            ),
        },
        "transition_factorization": {
            "target_state_weights": table_record(state_weights),
            "behavior_selected_state_weights": table_record(
                behavior_selected_weights
            ),
            "common_weight_causal_y1": fraction_record(
                common_weight_causal_y1
            ),
            "raw_row_factor_y1": fraction_record(row_factor_y1),
            "composition_factor_y1": fraction_record(composition_factor_y1),
            "product_y1": fraction_record(
                row_factor_y1 * composition_factor_y1
            ),
            "composition_level_all_outcomes": fraction_record(
                composition_level
            ),
            "raw_lambda_times_composition_level": fraction_record(
                conservative_product_level
            ),
        },
        "full_hidden_path_ratio_example": {
            "path": "S0=1,A0=1,Y=1,A1=1",
            "u1_over_u0_initial_factor": fraction_record(
                dgp.p_su_q[1][1] / dgp.p_su_q[1][0]
            ),
            "u1_over_u0_action0_factor": fraction_record(
                dgp.pi0_q[1][1] / dgp.pi0_q[1][0]
            ),
            "u1_over_u0_transition_factor": fraction_record(
                dgp.transition_y1_q[1][1][1]
                / dgp.transition_y1_q[1][1][0]
            ),
            "u1_over_u0_action1_factor": fraction_record(
                dgp.pi1_q / dgp.pi1_q
            ),
            "joint_path_ratio": fraction_record(path_ratio_example),
            "note": (
                "This hidden-path decomposition is distinct from Lambda_T, "
                "which compares interventional and observed transition rows."
            ),
        },
        "oracle_ipw_value": fraction_record(oracle_ipw),
        "checks": checks,
    }


def run() -> dict:
    return analyze(TransitionConfoundedH2DGP())


def render_report(result: dict) -> str:
    value = result["target_value"]["value"]
    levels = result["sensitivity_levels"]
    intervals = result["robust_value_intervals"]
    transition = result["transition_rows"]
    factors = result["transition_factorization"]
    checks = result["checks"]

    raw_level = levels["transition_raw_lambda_t"]["value"]
    latent_level = levels["transition_latent_lambda_t"]["value"]
    raw_interval = intervals["raw_at_raw_lambda_t"]
    latent_raw_interval = intervals["compressed_at_raw_lambda_t"]
    latent_corrected = intervals["compressed_at_required_lambda_t"]

    return rf"""# Exact transition-confounded H=2 stress test

## Headline

This finite population calculation exhibits genuine hidden transition
confounding:
\(T(Y\mid S_0,A_0,U)\) depends directly on \(U\).  The target
\(A_0=A_1=1\) has value \(V={value:.6f}\).

We use a multiplicative transition-row sensitivity level
\(\Lambda_T\), defined relative to the observed transition row.  This is
**not** the paper's action-odds \(\Gamma\).  The levels below are the
oracle-required population levels in this constructed DGP, not quantities
identified from logged data.

| Analyst state | \(\Lambda_T\) used | Robust value interval | Covers \(V\)? |
|---|---:|---:|---:|
| Raw \(S\) | {raw_level:.6f} | [{raw_interval['lower']['value']:.6f}, {raw_interval['upper']['value']:.6f}] | {checks['raw_interval_covers']} |
| Compressed \(Z\) | {raw_level:.6f} | [{latent_raw_interval['lower']['value']:.6f}, {latent_raw_interval['upper']['value']:.6f}] | {not checks['compressed_interval_at_raw_level_excludes']} |
| Compressed \(Z\) | {latent_level:.6f} | [{latent_corrected['lower']['value']:.6f}, {latent_corrected['upper']['value']:.6f}] | {checks['corrected_compressed_interval_covers']} |

The raw level is
\(\Lambda_{{T,S}}={levels['transition_raw_lambda_t']['fraction']}\)
\(={raw_level:.6f}\).  Compression raises the required level to
\(\Lambda_{{T,Z}}={levels['transition_latent_lambda_t']['fraction']}\)
\(={latent_level:.6f}\).  At the raw level, the compressed interval excludes
the truth from below; using the required compressed level restores coverage.
All intervals are nonvacuous.

## Why this isolates transition transport

The action-odds sensitivity moves in the opposite direction:

| Quantity | Raw | Compressed |
|---|---:|---:|
| Pairwise action-odds level | {levels['action_raw_pairwise_gamma']['value']:.6f} | {levels['action_latent_pairwise_gamma']['value']:.6f} |
| Transition-row level | {raw_level:.6f} | {latent_level:.6f} |

Thus the noncoverage cannot be explained by an increase in the paper's
action-odds \(\Gamma\): compression reduces that diagnostic while amplifying
the transition level.  Action selection still enters
\(T^{{\mathrm{{obs}}}}\), so \(\Lambda_T\) already absorbs that channel.

The exact transition probabilities for \(A_0=1\) are:

| State | Interventional \(T^{{\mathrm{{do}}}}(Y=1\mid s,1)\) | Observed \(T^{{\mathrm{{obs}}}}(Y=1\mid s,A_0=1)\) |
|---|---:|---:|
| \(s_1\) | {transition['causal_raw_y1'][0]['value']:.6f} | {transition['observed_raw_y1'][0]['value']:.6f} |
| \(s_2\) | {transition['causal_raw_y1'][1]['value']:.6f} | {transition['observed_raw_y1'][1]['value']:.6f} |
| merged \(z\) | {transition['causal_latent_y1']['value']:.6f} | {transition['observed_latent_y1']['value']:.6f} |

The second action is nontrivial because \(R=A_1Y\).  At fixed \(Y,A_1\), the
reward has no dependence on discarded state identity, so compression loses no
structural reward information.

## Transition factorization

Let \(\nu\) mix the raw interventional rows using the behavior-selected state
weights.  For outcome \(Y=1\),

\[
\frac{{T^{{\mathrm{{obs}}}}(1\mid z,A_0=1)}}{{T^{{\mathrm{{do}}}}(1\mid z,do(A_0=1))}}
=
\underbrace{{\frac{{T^{{\mathrm{{obs}}}}(1\mid z,A_0=1)}}{{\nu(1)}}}}_
{{{factors['raw_row_factor_y1']['value']:.6f}}}
\times
\underbrace{{\frac{{\nu(1)}}{{T^{{\mathrm{{do}}}}(1\mid z,do(A_0=1))}}}}_
{{{factors['composition_factor_y1']['value']:.6f}}}
=
{factors['product_y1']['value']:.6f}.
\]

The first factor is raw transition confounding under common weights; the
second is the shift from target state composition to behavior-selected state
composition after compression.  Across both binary outcomes, multiplying
the raw transition level by the composition level gives the conservative
level {factors['raw_lambda_times_composition_level']['value']:.6f}, whose
compressed interval also covers \(V\).

The JSON additionally records a full hidden-\(U\) path-ratio example.  That
path decomposition is reported separately and is not equated with
\(\Lambda_T\).

## Scope

This result answers a different question from the existing outcome-confounded
\(H=2\) analysis.  It shows that the representation-induced transport problem
persists when the transition kernel itself is confounded.  A complete general
trajectory theorem would need to combine action-policy, state-selection, and
transition-kernel factors under an explicitly stated joint sensitivity model.
This calculation does not silently multiply the paper's action \(\Gamma\) by
\(\Lambda_T\).  Its interval is sharp for the stated direct rectangular
transition-row model, not for a general hidden-\(U\) structural class or an
occupancy-ratio model.  The pairwise hidden-\(U\) transition level is a
different object and does not amplify here
({levels['transition_raw_pairwise_hidden_u']['value']:.1f} raw versus
{levels['transition_latent_pairwise_hidden_u']['value']:.1f} compressed).

## Exact checks

```text
{json.dumps(checks, indent=2, sort_keys=True)}
```
"""


def source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/exact_transition_h2"),
    )
    args = parser.parse_args()

    result = run()
    script_path = Path(__file__).resolve()
    dgp_path = (
        script_path.parents[1] / "dgp" / "transition_confounded_h2.py"
    )
    result["provenance"] = {
        "analysis_script": "experiments/exact_transition_h2.py",
        "analysis_script_sha256": source_sha256(script_path),
        "dgp_source": "dgp/transition_confounded_h2.py",
        "dgp_source_sha256": source_sha256(dgp_path),
        "arithmetic": "fractions.Fraction throughout; floats only for display",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "transition_confounded_h2_exact.json"
    report_path = args.output_dir / "transition_confounded_h2_exact.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report_path.write_text(render_report(result))

    print(render_report(result))
    print(f"\nJSON: {json_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
