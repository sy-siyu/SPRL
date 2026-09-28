"""Enumerate deterministic phi-measurable target policies in Appendix D.

For every deterministic policy on the two latent cells, lift the policy back
to the original state space and compare the same original-DGP value against
raw- and latent-space bounds.  Both the submitted Equation-11 calculation and
the normalized sharp one-step MSM calculation are reported.
"""

from __future__ import annotations

from itertools import product

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.exact_one_step import sharp_policy_bounds
from dgp.discrete_counterexample import build_counterexample
from estimators.scalar_cell_diagnostic import (
    compute_bounds_latent_space,
    compute_bounds_raw_state,
)


def covers(interval: tuple[float, float], value: float) -> bool:
    return interval[0] <= value <= interval[1]


def main() -> None:
    dgp = build_counterexample(gamma_s_target=2.0, alpha=0.95, q=0.9, r=0.1)
    gamma_s = dgp.compute_gamma_s()
    gamma_z = dgp.compute_gamma_z()
    state_to_z = np.array(
        [
            dgp.latent_indices.index(dgp.compression_map[state])
            for state in range(dgp.n_states)
        ],
        dtype=int,
    )

    print(
        "policy_z  policy_s  value   "
        "eq11_raw       eq11_latent@GS  sharp_raw      sharp_latent@GS  "
        "sharp_latent@GZ"
    )
    candidates = [
        (f"det-{''.join(str(int(x)) for x in actions)}", actions)
        for actions in product((0.0, 1.0), repeat=dgp.n_latent)
    ]
    candidates.extend(
        [
            ("random", (0.5, 0.5)),
            ("projected-greedy", (0.5, 0.0)),
            ("stochastic-25-75", (0.25, 0.75)),
            ("stochastic-75-25", (0.75, 0.25)),
        ]
    )

    for name, actions in candidates:
        pi_z = np.asarray(actions, dtype=float)
        pi_s = pi_z[state_to_z]
        value = dgp.true_value(pi_s)

        eq11_raw = compute_bounds_raw_state(dgp, pi_s, gamma_s)[:2]
        eq11_latent_s = compute_bounds_latent_space(dgp, pi_z, gamma_s)[:2]
        sharp_raw = sharp_policy_bounds(dgp, "raw", pi_s, gamma_s)
        sharp_latent_s = sharp_policy_bounds(dgp, "latent", pi_z, gamma_s)
        sharp_latent_z = sharp_policy_bounds(dgp, "latent", pi_z, gamma_z)

        def fmt(interval: tuple[float, float]) -> str:
            flag = "Y" if covers(interval, value) else "N"
            return f"[{interval[0]:.6f},{interval[1]:.6f}]/{flag}"

        print(
            f"{name:18s} {pi_z.tolist()}  {pi_s.tolist()}  "
            f"{value:.3f}  {fmt(eq11_raw):16s} {fmt(eq11_latent_s):17s} "
            f"{fmt(sharp_raw):16s} {fmt(sharp_latent_s):18s} "
            f"{fmt(sharp_latent_z)}"
        )


if __name__ == "__main__":
    main()
