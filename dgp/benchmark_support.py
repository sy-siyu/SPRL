"""IHDP source schema and exact finite-support observed propensity.

The source CSV is an external, user-supplied CEVAE IHDP asset. Its first five
columns are treatment/outcomes; columns 5:30 are the 25 covariates.
"""

from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from dgp.ihdp_bandit import IHDPConfoundedBandit


def load_ihdp_covariates(path: str | Path) -> np.ndarray:
    source = np.loadtxt(path, delimiter=",")
    if source.shape != (747, 30):
        raise ValueError(f"Expected 747 x 30 IHDP source, got {source.shape}")
    if not np.array_equal(np.unique(source[:, 0]), [0.0, 1.0]) or int(source[:, 0].sum()) != 139:
        raise ValueError("Unexpected IHDP treatment column")
    covariates = source[:, 5:30]
    return StandardScaler().fit_transform(covariates)


class EmpiricalShiftSupport:
    """Exact P(U=1|S) under uniform empirical resampling and a U shift."""

    def __init__(self, x: np.ndarray, n_instrument: int = 10, shift: float = 2.0):
        unshifted = np.ascontiguousarray(x)
        shifted = unshifted.copy()
        shifted[:, :n_instrument] += shift
        key = lambda row: np.ascontiguousarray(row).tobytes()
        self.count0 = Counter(map(key, unshifted))
        self.count1 = Counter(map(key, shifted))
        overlap = self.count0.keys() & self.count1.keys()
        self.overlap_points = len(overlap)
        self.overlap_mass0 = sum(self.count0[k] for k in overlap)
        self.overlap_mass1 = sum(self.count1[k] for k in overlap)

    def posterior_u1(self, s: np.ndarray) -> np.ndarray:
        q = np.empty(len(s), dtype=float)
        for i, row in enumerate(s):
            key = np.ascontiguousarray(row).tobytes()
            n0, n1 = self.count0[key], self.count1[key]
            if n0 + n1 == 0:
                raise ValueError(f"Observed state {i} is outside the empirical shifted support")
            q[i] = n1 / (n0 + n1)
        return q

    def propensity(self, dgp: IHDPConfoundedBandit, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = self.posterior_u1(s)
        linear = s @ dgp.w_s + dgp.bias
        p0 = 1.0 / (1.0 + np.exp(-np.clip(linear, -500, 500)))
        p1 = 1.0 / (1.0 + np.exp(-np.clip(linear + dgp.w_u, -500, 500)))
        return (1 - q) * p0 + q * p1, q


def sample_ihdp_with_subjects(dgp: IHDPConfoundedBandit, n: int,
                              support: EmpiricalShiftSupport) -> dict:
    """Follow the original DGP RNG order while keeping sampled subject IDs."""
    subject = dgp.rng.choice(dgp.N, size=n, replace=True)
    s = dgp.X_base[subject].copy()
    u = dgp.rng.binomial(1, 0.5, size=n).astype(float)
    for dim in dgp.idx_instrument:
        s[:, dim] += u * dgp.conf_strength
    logits = s @ dgp.w_s + dgp.w_u * u + dgp.bias
    pi_b = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
    a = dgp.rng.binomial(1, pi_b).astype(float)
    r = s @ dgp.w_r + dgp.w_ra * a + dgp.rng.randn(n) * dgp.reward_noise
    pi_obs, posterior = support.propensity(dgp, s)
    return {"s": s, "a": a, "r": r, "u": u, "subject": subject,
            "pi_b_conditional": pi_b, "pi_b_observed": pi_obs,
            "posterior_u1": posterior}


def analytic_ihdp_target(dgp: IHDPConfoundedBandit) -> float:
    return float(dgp.X_base.mean(axis=0) @ dgp.w_r + dgp.w_ra)
