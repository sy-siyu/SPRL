"""
Statewise scalar HT-IPW calculation retained as an appendix diagnostic.

This collapses each observed reward law to its mean and varies a single
propensity per cell. It is NOT a valid general full-MSM bound. For the
normalized reward-distribution LP use experiments/exact_one_step.py; for
Tables 1 and IHDP use robust_ipw_continuous.scalar_cell_bounds_samplewise.

For each state z with observed propensity e(z) = P(A=1|Z=z), the MSM
at level Γ constrains the unknown true propensity e*(z) to:

    e*(z) ∈ [e(z) / (e(z) + Γ(1-e(z))),  Γe(z) / (Γe(z) + (1-e(z)))]

The HT-IPW policy value for a given e* assignment:

    V(π_e; {e*}) = Σ_z P(z) [π_e(1|z) · e(z) · r_obs(z,1) / e*(z)
                             + π_e(0|z) · (1-e(z)) · r_obs(z,0) / (1-e*(z))]

The bounds optimize over all valid e* assignments.e
"""

import numpy as np
from scipy.optimize import minimize_scalar


def msm_propensity_bounds(e, gamma):
    """
    Compute the MSM sensitivity set for the true propensity e*.

    Parameters
    ----------
    e : float
        Observed marginal propensity P(A=1|Z=z).
    gamma : float
        Sensitivity parameter Γ ≥ 1.

    Returns
    -------
    e_low, e_high : float
        Bounds on the true propensity.
    """
    e = np.clip(e, 1e-12, 1 - 1e-12)
    e_low = e / (e + gamma * (1 - e))
    e_high = gamma * e / (gamma * e + (1 - e))
    return e_low, e_high


def ht_ipw_value_per_state(pi_e_1, e_obs, r_obs_1, r_obs_0, e_star):
    """
    Compute HT-IPW value contribution from a single state z.

    V_z(e*) = π_e(1|z) · e(z) · r_obs(z,1) / e*
            + π_e(0|z) · (1-e(z)) · r_obs(z,0) / (1-e*)

    Parameters
    ----------
    pi_e_1 : float
        Target policy π_e(A=1|z).
    e_obs : float
        Observed propensity e(z) = P(A=1|Z=z).
    r_obs_1, r_obs_0 : float
        Observational conditional rewards E[R|Z=z,A=a].
    e_star : float
        Hypothesized true propensity.
    """
    e_star = np.clip(e_star, 1e-12, 1 - 1e-12)
    term1 = pi_e_1 * e_obs * r_obs_1 / e_star
    term0 = (1 - pi_e_1) * (1 - e_obs) * r_obs_0 / (1 - e_star)
    return term1 + term0


def scalar_cell_bounds(pi_e, pi_b_marg, r_obs, p_states, gamma):
    """
    Compute scalar HT-IPW endpoints on V(π_e) using MSM at level Γ.

    For each state z, independently optimize over e*(z) ∈ MSM set(z, Γ).

    Parameters
    ----------
    pi_e : array (n_states,)
        Target policy π_e(A=1|z) for each state.
    pi_b_marg : array (n_states,)
        Observed marginal propensity e(z) = P(A=1|Z=z).
    r_obs : array (n_states, 2)
        Observational conditional rewards E[R|Z=z, A=a].
    p_states : array (n_states,)
        Marginal state distribution P(Z=z).
    gamma : float
        Sensitivity parameter Γ ≥ 1.

    Returns
    -------
    v_min : float
        Lower bound on V(π_e).
    v_max : float
        Upper bound on V(π_e).
    info : dict
        Per-state bounds and optimal e* values.
    """
    n_states = len(pi_e)
    v_max_total = 0.0
    v_min_total = 0.0

    info = {'per_state': []}

    for z in range(n_states):
        e = pi_b_marg[z]
        e_low, e_high = msm_propensity_bounds(e, gamma)

        def neg_f(e_star):
            return -ht_ipw_value_per_state(
                pi_e[z], e, r_obs[z, 1], r_obs[z, 0], e_star
            )

        def pos_f(e_star):
            return ht_ipw_value_per_state(
                pi_e[z], e, r_obs[z, 1], r_obs[z, 0], e_star
            )

        # Optimize over e* in [e_low, e_high]
        res_max = minimize_scalar(neg_f, bounds=(e_low, e_high), method='bounded')
        res_min = minimize_scalar(pos_f, bounds=(e_low, e_high), method='bounded')

        f_max = -res_max.fun
        f_min = res_min.fun
        e_star_max = res_max.x
        e_star_min = res_min.x

        v_max_total += p_states[z] * f_max
        v_min_total += p_states[z] * f_min

        info['per_state'].append({
            'z': z,
            'e_obs': e,
            'e_low': e_low,
            'e_high': e_high,
            'e_star_max': e_star_max,
            'e_star_min': e_star_min,
            'f_max': f_max,
            'f_min': f_min,
        })

    return v_min_total, v_max_total, info


def compute_bounds_raw_state(dgp, pi_e_s, gamma):
    """
    Compute scalar HT-IPW endpoints in the raw state space S.

    Parameters
    ----------
    dgp : DiscreteCounterexampleDGP
        The data generating process.
    pi_e_s : array (n_states,)
        Target policy π_e(A=1|S=s).
    gamma : float
        Sensitivity parameter Γ.

    Returns
    -------
    v_min, v_max, info
    """
    return scalar_cell_bounds(
        pi_e=pi_e_s,
        pi_b_marg=dgp.pi_b_marg,
        r_obs=dgp.rewards,  # E[R|S=s,A=a] = r(s,a) (no bias at S level)
        p_states=dgp.p_s,
        gamma=gamma,
    )


def compute_bounds_latent_space(dgp, pi_e_z, gamma):
    """
    Compute scalar HT-IPW endpoints in the latent space Z.

    Parameters
    ----------
    dgp : DiscreteCounterexampleDGP
        The data generating process.
    pi_e_z : array (n_latent,)
        Target policy π_e(A=1|Z=z).
    gamma : float
        Sensitivity parameter Γ.

    Returns
    -------
    v_min, v_max, info
    """
    return scalar_cell_bounds(
        pi_e=pi_e_z,
        pi_b_marg=dgp.pi_b_z_marg,
        r_obs=dgp.rewards_z,  # E[R|Z=z,A=a] (observational, includes bias)
        p_states=dgp.p_z,
        gamma=gamma,
    )
