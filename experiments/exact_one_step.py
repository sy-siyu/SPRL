"""Recompute Appendix D with coarse Eq. 11 and sharp one-step MSM bounds.

The sharp calculation uses outcome-dependent likelihood-ratio weights within
each observed (state, action) cell.  The weights satisfy the standard binary
marginal-sensitivity-model box constraints and the normalization constraint

    E[w_a(Y) | X=x, A=a] = 1.

The scalar cell calculation is included for comparison only; the normalized
LP retains the observed reward distribution.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dgp.discrete_counterexample import build_counterexample
from estimators.scalar_cell_diagnostic import (
    compute_bounds_latent_space,
    compute_bounds_raw_state,
)


def _weight_bounds(e_obs: float, action: int, gamma: float) -> tuple[float, float]:
    """Return MSM bounds for p(A=a|X)/p(A=a|X,U)."""
    if action == 1:
        return (
            e_obs + (1.0 - e_obs) / gamma,
            e_obs + gamma * (1.0 - e_obs),
        )
    return (
        (1.0 - e_obs) + e_obs / gamma,
        (1.0 - e_obs) + gamma * e_obs,
    )


def _observed_reward_distribution(dgp, state_space: str, x: int, action: int):
    """Return reward values and P(R=r | X=x,A=a) for the exact DGP."""
    masses: dict[float, float] = {}
    for s in range(dgp.n_states):
        if state_space == "raw":
            belongs = s == x
        elif state_space == "latent":
            belongs = dgp.latent_indices.index(dgp.compression_map[s]) == x
        else:
            raise ValueError(f"Unknown state space: {state_space}")
        if not belongs:
            continue

        for u in range(dgp.n_u):
            p_action = dgp.pi_b_full[s, u, action]
            mass = dgp.p_su[s, u] * p_action
            reward = float(dgp.rewards[s, action])
            masses[reward] = masses.get(reward, 0.0) + mass

    total = sum(masses.values())
    rewards = np.array(sorted(masses), dtype=float)
    probabilities = np.array([masses[y] / total for y in rewards], dtype=float)
    return rewards, probabilities


def _sharp_cell_mean(
    rewards: np.ndarray,
    probabilities: np.ndarray,
    e_obs: float,
    action: int,
    gamma: float,
) -> tuple[float, float]:
    """Solve the normalized outcome-dependent MSM LP in one observed cell."""
    lower, upper = _weight_bounds(e_obs, action, gamma)
    objective = probabilities * rewards
    equality_matrix = probabilities[None, :]
    equality_target = np.array([1.0])
    bounds = [(lower, upper)] * len(rewards)

    minimum = linprog(
        objective,
        A_eq=equality_matrix,
        b_eq=equality_target,
        bounds=bounds,
        method="highs",
    )
    maximum = linprog(
        -objective,
        A_eq=equality_matrix,
        b_eq=equality_target,
        bounds=bounds,
        method="highs",
    )
    if not minimum.success or not maximum.success:
        raise RuntimeError((minimum.message, maximum.message))
    return float(minimum.fun), float(-maximum.fun)


def sharp_policy_bounds(dgp, state_space: str, pi_e, gamma: float):
    """Compute sharp one-step MSM policy-value bounds."""
    if state_space == "raw":
        state_masses = dgp.p_s
        propensities = dgp.pi_b_marg
    elif state_space == "latent":
        state_masses = dgp.p_z
        propensities = dgp.pi_b_z_marg
    else:
        raise ValueError(f"Unknown state space: {state_space}")

    lower_total = 0.0
    upper_total = 0.0
    for x, p_x in enumerate(state_masses):
        for action in (0, 1):
            target_probability = pi_e[x] if action == 1 else 1.0 - pi_e[x]
            if target_probability == 0.0:
                continue
            rewards, probabilities = _observed_reward_distribution(
                dgp, state_space, x, action
            )
            cell_lower, cell_upper = _sharp_cell_mean(
                rewards, probabilities, propensities[x], action, gamma
            )
            lower_total += p_x * target_probability * cell_lower
            upper_total += p_x * target_probability * cell_upper
    return lower_total, upper_total


def main() -> None:
    dgp = build_counterexample(gamma_s_target=2.0, alpha=0.95, q=0.9, r=0.1)
    gamma_s = dgp.compute_gamma_s()
    gamma_z = dgp.compute_gamma_z()
    pi_e_s = np.ones(dgp.n_states)
    pi_e_z = np.ones(dgp.n_latent)
    true_value = dgp.true_value(pi_e_s)

    eq11_raw = compute_bounds_raw_state(dgp, pi_e_s, gamma_s)[:2]
    eq11_latent_s = compute_bounds_latent_space(dgp, pi_e_z, gamma_s)[:2]
    eq11_latent_z = compute_bounds_latent_space(dgp, pi_e_z, gamma_z)[:2]

    sharp_raw = sharp_policy_bounds(dgp, "raw", pi_e_s, gamma_s)
    sharp_latent_s = sharp_policy_bounds(dgp, "latent", pi_e_z, gamma_s)
    sharp_latent_z = sharp_policy_bounds(dgp, "latent", pi_e_z, gamma_z)

    # Appendix D discards reward-relevant S.  Hence the hidden variable needed
    # for latent ignorability is H=(S,U), not the original U alone.
    z1_states = [
        s
        for s in range(dgp.n_states)
        if dgp.latent_indices.index(dgp.compression_map[s]) == 0
    ]
    latent_odds = dgp.pi_b_z_marg[0] / (1.0 - dgp.pi_b_z_marg[0])
    full_hidden_odds = [
        dgp.pi_b[s, u] / (1.0 - dgp.pi_b[s, u])
        for s in z1_states
        for u in range(dgp.n_u)
    ]
    full_hidden_nominal_gamma = max(
        max(odds / latent_odds, latent_odds / odds)
        for odds in full_hidden_odds
    )
    full_hidden_pairwise_gamma = max(full_hidden_odds) / min(full_hidden_odds)

    print(f"Gamma_S: {gamma_s:.12f}")
    print(f"Paper's original-U pairwise Gamma_Z: {gamma_z:.12f}")
    print(
        "Full H=(S,U) latent nominal/pairwise Gamma: "
        f"{full_hidden_nominal_gamma:.12f} / {full_hidden_pairwise_gamma:.12f}"
    )
    print(f"True value: {true_value:.12f}")
    print(f"Eq. 11 raw at Gamma_S:       {eq11_raw}")
    print(f"Sharp raw at Gamma_S:        {sharp_raw}")
    print(f"Eq. 11 latent at Gamma_S:    {eq11_latent_s}")
    print(f"Sharp latent at Gamma_S:     {sharp_latent_s}")
    print(f"Eq. 11 latent at Gamma_Z:    {eq11_latent_z}")
    print(f"Sharp latent at Gamma_Z:     {sharp_latent_z}")


if __name__ == "__main__":
    main()
