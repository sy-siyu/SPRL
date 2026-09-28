"""Sign-aware samplewise HT-IPW outer envelopes for binary actions.

The numerical ``gamma`` argument denotes the marginal sensitivity input
Lambda. These empirical plug-in endpoints are not confidence intervals.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


def msm_propensity_bounds(e, gamma):
    """Return propensity endpoints for an odds deviation of at most Lambda."""
    if not np.isfinite(gamma) or gamma < 1:
        raise ValueError("The marginal sensitivity input must be finite and >= 1")
    e = np.clip(e, 1e-8, 1 - 1e-8)
    e_low = e / (e + gamma * (1 - e))
    e_high = gamma * e / (gamma * e + (1 - e))
    return e_low, e_high


def estimate_propensity(z, a, method="logistic"):
    """Fit the in-sample logistic readout used in Setting A."""
    if method != "logistic":
        raise ValueError(f"Unknown method: {method}")
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(z, a)
    e_hat = model.predict_proba(z)[:, 1]
    return np.clip(e_hat, 1e-8, 1 - 1e-8), model


def robust_ipw_bounds_samplewise(a, r, pi_e_a, gamma, e_hat):
    """Evaluate pointwise denominator endpoints before averaging rewards.

    ``pi_e_a`` is the target-policy probability of each observed action;
    ``e_hat`` is P(A=1|X), either known or fitted. Bounds are deliberately
    unnormalized: they are a conservative population envelope when the
    propensity and sensitivity premises hold, not the sharp normalized MSM
    solution. An estimated representation-level propensity does not itself
    verify those premises.
    """
    arrays = [np.asarray(x) for x in (a, r, pi_e_a, e_hat)]
    a, r, pi_e_a, e_hat = arrays
    if not arrays[0].size or any(x.ndim != 1 or x.shape != a.shape for x in arrays):
        raise ValueError("Inputs must be nonempty one-dimensional arrays of equal length")
    if any(not np.isfinite(x).all() for x in arrays):
        raise ValueError("Inputs must be finite")
    if not np.isin(a, [0, 1]).all():
        raise ValueError("Actions must be binary")
    if ((pi_e_a < 0) | (pi_e_a > 1)).any() or ((e_hat < 0) | (e_hat > 1)).any():
        raise ValueError("Probabilities must lie in [0,1]")
    e_hat = np.clip(e_hat, 1e-8, 1 - 1e-8)
    e_low, e_high = msm_propensity_bounds(e_hat, gamma)
    denom_low = np.where(a == 1, e_low, 1.0 - e_high)
    denom_high = np.where(a == 1, e_high, 1.0 - e_low)
    numer = pi_e_a * r
    contrib_min = np.where(numer >= 0, numer / denom_high, numer / denom_low)
    contrib_max = np.where(numer >= 0, numer / denom_low, numer / denom_high)
    info = {"mode": "samplewise", "e_hat_mean": float(np.mean(e_hat)),
            "n_samples": int(len(a))}
    return float(np.mean(contrib_min)), float(np.mean(contrib_max)), info
