"""
Semi-synthetic one-step bandit using IHDP covariates.

S = real IHDP covariates (25-dim), resampled with replacement.
U ~ Bernoulli(0.5), injected.
U shifts a subset of covariates (S_a dims) — confounded, NOT reward-relevant.
pi_b(1|s,u) = sigma(w_s^T s + w_u * u + bias)
R = w_r^T s + w_ra * a + noise
Reward depends on S_r dims only (NOT S_a).

Confounding and reward features occupy disjoint coordinates. Original IHDP
treatments and outcomes are excluded from this semi-synthetic experiment.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


def load_ihdp_covariates(csv_path='data/ihdp_npci_1.csv'):
    """
    Load IHDP covariates (25-dim) from CSV.

    CEVAE CSV format: columns 0:5 are treatment/outcomes; columns 5:30
    are the 25 covariates. Only covariates are used.

    Returns
    -------
    X : array (N, 25), standardised covariates
    """
    data = np.loadtxt(csv_path, delimiter=',')
    if data.shape != (747, 30):
        raise ValueError(f"Expected the 747-by-30 CEVAE IHDP source, got {data.shape}")
    X = data[:, 5:30]
    # Standardise each covariate to mean=0, std=1
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X


class IHDPConfoundedBandit:
    """
    Semi-synthetic one-step bandit using IHDP covariates.

    Parameters
    ----------
    X_ihdp : array (N, 25)
        Standardised IHDP covariates.
    n_instrument : int
        Number of dims U shifts (S_a group). Default 5.
    w_u : float
        Confounding strength -> Gamma_S = exp(|w_u|).
    conf_strength : float
        Magnitude of U -> S shift (in std units).
    policy_clean_weight : float
        Strength of policy weights on clean (reward-relevant) dims.
        Aligns policy with reward direction for strong Setting A signal.
    seed : int or None
    """

    def __init__(self, X_ihdp, n_instrument=10, w_u=0.5,
                 conf_strength=2.0, policy_clean_weight=0.6, seed=None):
        self.X_base = X_ihdp.copy()
        self.N = X_ihdp.shape[0]
        self.state_dim = 25
        self.rng = np.random.RandomState(seed)

        # Feature groups
        # S_a: first n_instrument dims shifted by U (confounded, NOT reward-relevant)
        # S_r: remaining dims (reward-relevant, NOT confounded)
        self.n_instrument = n_instrument
        self.idx_instrument = list(range(n_instrument))
        self.idx_clean = list(range(n_instrument, self.state_dim))

        # Confounding: U shifts S_a dims
        self.conf_strength = conf_strength

        # Behavioral policy: pi_b(1|s,u) = sigma(w_s^T s + w_u * u + bias)
        self.w_s = self.rng.randn(self.state_dim) * 0.1  # small random weights
        self.w_u = w_u
        self.bias = 0.0

        # Reward: depends on S_r dims only (NOT S_a)
        self.w_r = np.zeros(self.state_dim)
        reward_dir = self.rng.randn(len(self.idx_clean))
        reward_dir /= np.linalg.norm(reward_dir) + 1e-8
        for i, idx in enumerate(self.idx_clean):
            self.w_r[idx] = reward_dir[i] * 0.5

        # Align policy weights on clean dims with reward direction
        # Synthetic policy-reward alignment in the stated experimental design.
        if policy_clean_weight > 0:
            for i, idx in enumerate(self.idx_clean):
                self.w_s[idx] = reward_dir[i] * policy_clean_weight

        self.w_ra = 0.3  # action effect on reward
        self.reward_noise = 0.1

    @property
    def gamma_s(self):
        return np.exp(abs(self.w_u))

    def compute_gamma_s(self, n_samples=100000):
        """Estimate Gamma_S numerically from large state sample."""
        max_ratio = 1.0
        for _ in range(n_samples):
            idx = self.rng.randint(self.N)
            s = self.X_base[idx].copy()
            u_val = self.rng.binomial(1, 0.5)
            # Apply U -> S shift
            for dim in self.idx_instrument:
                s[dim] += u_val * self.conf_strength

            logit0 = np.dot(s, self.w_s) + self.w_u * 0 + self.bias
            logit1 = np.dot(s, self.w_s) + self.w_u * 1 + self.bias

            p0 = np.clip(1.0 / (1.0 + np.exp(-logit0)), 0.001, 0.999)
            p1 = np.clip(1.0 / (1.0 + np.exp(-logit1)), 0.001, 0.999)

            odds0 = p0 / (1 - p0)
            odds1 = p1 / (1 - p1)
            ratio = max(odds1 / odds0, odds0 / odds1)
            max_ratio = max(max_ratio, ratio)

        return max_ratio

    def sample(self, n_samples):
        """Sample n_samples one-step transitions."""
        # Sample subjects with replacement
        indices = self.rng.choice(self.N, size=n_samples, replace=True)
        s_base = self.X_base[indices].copy()

        # Sample U
        u = self.rng.binomial(1, 0.5, size=n_samples).astype(float)

        # U -> S: shift S_a dims
        for dim in self.idx_instrument:
            s_base[:, dim] += u * self.conf_strength
        s = s_base

        # U -> A: confounded behavioral policy
        logits = s @ self.w_s + self.w_u * u + self.bias
        pi_b = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
        a = self.rng.binomial(1, pi_b).astype(float)

        # Conditional mixture weights use the exact shifted empirical support,
        # rather than the unconditional Bernoulli prior on U.
        from dgp.benchmark_support import EmpiricalShiftSupport
        support = EmpiricalShiftSupport(self.X_base, self.n_instrument, self.conf_strength)
        pi_b_marg, _ = support.propensity(self, s)

        # Reward: depends on S_r only
        r = s @ self.w_r + self.w_ra * a + self.rng.randn(n_samples) * self.reward_noise

        return {
            's': s,
            'a': a,
            'r': r,
            'u': u,
            'pi_b': pi_b,
            'pi_b_marg': pi_b_marg,
        }

    def true_value(self, pi_e=1, n_mc=100000):
        """V(pi_e) via Monte Carlo. Default: pi_e always takes action 1."""
        data = self.sample(n_mc)
        r_under_pi_e = data['s'] @ self.w_r + self.w_ra * pi_e
        return float(r_under_pi_e.mean())
