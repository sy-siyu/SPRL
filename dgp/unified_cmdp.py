"""
Unified confounded MDP with explicit four-group state structure.

State groups:
  S_∅ (noise):       Not shifted by U, not in dynamics/rewards. Should be discarded.
  S_a (instruments):  Shifted by U, NOT in dynamics/rewards. Pure confounding features.
  S_Δ (confounded):   Shifted by U AND in dynamics/rewards. The hard case.
  S_r (clean):        Not shifted by U, in dynamics/rewards. Safe to capture.

Setting A = S_Δ empty (clean separation)
Setting B = S_Δ non-empty (overlap)

Causal structure:
  U ~ Bernoulli(0.5), sampled ONCE per trajectory
  S_0 = [S_∅ ~ N(0,I), S_a ~ N(cs*(2U-1), I), S_Δ ~ N(cs*(2U-1), I), S_r ~ N(0,I)]
  A_t | S_t, U ~ Bernoulli(σ(w_s^T S_t + w_u * U + bias))
  S_{t+1,dynamics} = W @ S_{t,dynamics} + w_action * A_t + noise  (dynamics on S_Δ ∪ S_r only)
  S_{t+1,∅} ~ N(0,I),  S_{t+1,a} ~ N(cs*(2U-1), I)  (re-sampled, not dynamic)
  R_t = w_r^T S_{t,dynamics} + w_ra * A_t + noise

Γ_S = exp(|w_u|), per-step sensitivity parameter.
"""

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class UnifiedCMDP:
    """
    Unified confounded MDP with explicit four-group state structure.

    Parameters
    ----------
    dim_noise : int
        |S_∅|, noise dimensions (not shifted, not in dynamics)
    dim_instrument : int
        |S_a|, action instruments (U-shifted, not in dynamics)
    dim_confounded : int
        |S_Δ|, confounded dynamics (U-shifted AND in dynamics)
    dim_clean : int
        |S_r|, clean dynamics (not U-shifted, in dynamics)
    conf_strength : float
        How much U shifts S_a and S_Δ
    w_u : float
        Behavioral policy confounding weight. Γ_S = exp(|w_u|)
    horizon : int
        H=1 for Setting A, H>1 for Setting B
    transition_noise : float
        Std of Gaussian noise in transitions
    reward_noise_std : float
        Std of Gaussian noise in rewards
    seed : int or None
        Random seed for weight initialization
    """

    def __init__(self, dim_noise=2, dim_instrument=2, dim_confounded=0,
                 dim_clean=5, conf_strength=2.0, w_u=0.5, horizon=1,
                 transition_noise=0.1, reward_noise_std=0.1,
                 policy_conf_weight=None, policy_clean_weight=None,
                 seed=None):

        rng = np.random.RandomState(seed)

        self.dim_noise = dim_noise
        self.dim_instrument = dim_instrument
        self.dim_confounded = dim_confounded
        self.dim_clean = dim_clean
        self.state_dim = dim_noise + dim_instrument + dim_confounded + dim_clean
        self.conf_strength = conf_strength
        self.w_u = w_u
        self.horizon = horizon
        self.transition_noise = transition_noise
        self.reward_noise_std = reward_noise_std

        # Feature group indices
        idx = 0
        self.idx_noise = list(range(idx, idx + dim_noise)); idx += dim_noise
        self.idx_instrument = list(range(idx, idx + dim_instrument)); idx += dim_instrument
        self.idx_confounded = list(range(idx, idx + dim_confounded)); idx += dim_confounded
        self.idx_clean = list(range(idx, idx + dim_clean)); idx += dim_clean

        # U-shifted dims = instruments + confounded
        self.idx_u_shifted = self.idx_instrument + self.idx_confounded
        # Dynamics-relevant dims = confounded + clean
        self.idx_dynamics = self.idx_confounded + self.idx_clean
        self.dim_dynamics = len(self.idx_dynamics)

        # Confounding shift vector: nonzero on U-shifted dims
        self.w_conf = np.zeros(self.state_dim)
        if len(self.idx_u_shifted) > 0:
            self.w_conf[self.idx_u_shifted] = conf_strength / np.sqrt(len(self.idx_u_shifted))

        # Behavioral policy weights on state.
        # IMPORTANT: w_s must be generated BEFORE w_r to preserve rng order
        # from the original DGP (backward compatibility).
        #
        # Two styles:
        #   None/None (Setting B default): random weights, original multi-step DGP
        #   Explicit values (Setting A): structured weights aligned with confounding/reward
        if policy_conf_weight is None and policy_clean_weight is None:
            # Random policy weights (Setting B default / backward compatible)
            self.w_s = rng.randn(self.state_dim) * 0.3
            self.bias = -0.3
            self._policy_style = 'random'
        else:
            # Placeholder — will be filled after w_r is generated (needs reward direction)
            self.w_s = None
            self.bias = 0.0
            self._policy_style = 'aligned'

        # Reward weights: nonzero on dynamics-relevant dims ONLY
        self.w_r = np.zeros(self.state_dim)
        if self.dim_dynamics > 0:
            self.w_r[self.idx_dynamics] = rng.randn(self.dim_dynamics) * 0.5
        elif self.dim_clean > 0:
            self.w_r[self.idx_clean] = rng.randn(self.dim_clean) * 0.5
        self.w_ra = 0.2  # action effect on reward
        self.r_bias = 0.5

        # Now fill aligned policy weights if needed (Setting A)
        if self._policy_style == 'aligned':
            pcw = policy_conf_weight if policy_conf_weight is not None else 0.5
            pclw = policy_clean_weight if policy_clean_weight is not None else 0.45
            self.w_s = np.zeros(self.state_dim)

            # Policy weight on U-shifted dims (instruments + confounded)
            if len(self.idx_u_shifted) > 0 and pcw > 0:
                conf_unit = self.w_conf[self.idx_u_shifted].copy()
                norm = np.linalg.norm(conf_unit)
                if norm > 1e-8:
                    conf_unit /= norm
                self.w_s[self.idx_u_shifted] = pcw * conf_unit

            # Policy weight on reward-relevant dims, aligned with reward direction
            reward_dims = self.idx_dynamics if self.dim_dynamics > 0 else self.idx_clean
            if len(reward_dims) > 0 and pclw > 0:
                r_on_reward = self.w_r[reward_dims].copy()
                norm = np.linalg.norm(r_on_reward)
                if norm > 1e-8:
                    r_on_reward /= norm
                self.w_s[reward_dims] = pclw * r_on_reward

        # Transition matrix (for Setting B, H > 1)
        if horizon > 1 and self.dim_dynamics > 0:
            dd = self.dim_dynamics
            W = rng.randn(dd, dd) * 0.3
            # Ensure stability: spectral radius < 0.9
            max_eig = np.max(np.abs(np.linalg.eigvals(W)))
            if max_eig > 0:
                W *= 0.8 / max_eig
            self.W_trans = W
            # Action effect on dynamics dims
            self.w_action_dyn = rng.randn(dd) * 0.3
        else:
            self.W_trans = None
            self.w_action_dyn = None

        # Gamma_S
        self._gamma_s = np.exp(np.abs(self.w_u))

    @property
    def gamma_s(self):
        return self._gamma_s

    def _extract_dynamics(self, s):
        """Extract dynamics-relevant dims from state."""
        s = np.atleast_2d(s)
        return s[:, self.idx_dynamics]

    def _set_dynamics(self, s, s_dyn):
        """Set dynamics-relevant dims in state."""
        s = s.copy()
        s[:, self.idx_dynamics] = s_dyn
        return s

    def pi_b(self, s, u):
        """Behavioral policy P(A=1|S=s, U=u). s: (n, d) or (d,)."""
        s = np.atleast_2d(s)
        logit = s @ self.w_s + self.w_u * u + self.bias
        return _sigmoid(logit).ravel()

    def p_u_given_s(self, s):
        """Posterior P(U=1|S) via Bayes' rule with isotropic noise."""
        s = np.atleast_2d(s)
        # Only U-shifted dims are informative
        diff = 2.0 * (s @ self.w_conf)
        return _sigmoid(diff).ravel()

    def pi_b_marg(self, s):
        """Marginal propensity P(A=1|S) = E_U[π_b(A=1|S,U) | S]."""
        s = np.atleast_2d(s)
        p_u1 = self.p_u_given_s(s)
        return (p_u1 * self.pi_b(s, 1) + (1.0 - p_u1) * self.pi_b(s, 0)).ravel()

    def reward_fn(self, s, a):
        """Expected reward. Only depends on dynamics-relevant dims."""
        s = np.atleast_2d(s)
        return (s @ self.w_r + self.w_ra * a + self.r_bias).ravel()

    def transition(self, s, a, u, rng):
        """
        S_{t+1}: dynamics dims follow linear transition,
        noise/instrument dims are re-sampled.
        """
        s = np.atleast_2d(s)
        a = np.atleast_1d(a)
        n = len(s)

        s_next = np.zeros_like(s)

        # Noise dims: re-sample
        if self.dim_noise > 0:
            s_next[:, self.idx_noise] = rng.randn(n, self.dim_noise)

        # Instrument dims: re-sample with U shift
        if self.dim_instrument > 0:
            shift = np.outer(2 * u - 1, self.w_conf[self.idx_instrument])
            s_next[:, self.idx_instrument] = rng.randn(n, self.dim_instrument) + shift

        # Dynamics dims: linear transition
        if self.dim_dynamics > 0:
            s_dyn = self._extract_dynamics(s)
            s_dyn_next = s_dyn @ self.W_trans.T + np.outer(a, self.w_action_dyn)
            s_dyn_next += rng.randn(n, self.dim_dynamics) * self.transition_noise
            s_next[:, self.idx_dynamics] = s_dyn_next

        return s_next

    def sample_trajectories(self, n_trajectories, rng=None):
        """
        Sample n_trajectories of length H.

        Returns dict with:
          s: (n, H, d) — states
          a: (n, H) — actions
          r: (n, H) — rewards
          s_next: (n, H, d) — next states
          u: (n,) — confounder (one per trajectory)
          pi_b_marg: (n, H) — marginal propensity for observed action
        """
        if rng is None:
            rng = np.random.RandomState()

        n = n_trajectories
        H = self.horizon
        d = self.state_dim

        u = rng.binomial(1, 0.5, size=n).astype(float)

        all_s = np.zeros((n, H, d))
        all_a = np.zeros((n, H))
        all_r = np.zeros((n, H))
        all_s_next = np.zeros((n, H, d))
        all_pi_b_marg = np.zeros((n, H))

        # Initial state
        s_t = rng.randn(n, d)
        # Shift U-affected dims
        s_t += np.outer(2 * u - 1, self.w_conf)

        for t in range(H):
            all_s[:, t, :] = s_t

            # Action
            prob_a1 = self.pi_b(s_t, u)
            a_t = rng.binomial(1, prob_a1).astype(float)
            all_a[:, t] = a_t

            # Marginal propensity for observed action
            marg_1 = self.pi_b_marg(s_t)
            all_pi_b_marg[:, t] = np.where(a_t == 1, marg_1, 1 - marg_1)

            # Reward
            r_mean = self.reward_fn(s_t, a_t)
            all_r[:, t] = r_mean + rng.randn(n) * self.reward_noise_std

            # Transition
            s_next = self.transition(s_t, a_t, u, rng)
            all_s_next[:, t, :] = s_next
            s_t = s_next

        return {
            's': all_s,
            'a': all_a,
            'r': all_r,
            's_next': all_s_next,
            'u': u,
            'pi_b_marg': all_pi_b_marg,
        }

    def sample_one_step(self, n_samples, rng=None):
        """
        Sample one-step data (for Setting A, H=1).

        Returns dict with:
          s, a, r, u, pi_b_marg — all arrays of length n_samples
        """
        if rng is None:
            rng = np.random.RandomState()

        n = n_samples
        d = self.state_dim

        u = rng.binomial(1, 0.5, size=n).astype(float)
        s = rng.randn(n, d) + np.outer(2 * u - 1, self.w_conf)

        prob_a1 = self.pi_b(s, u)
        a = rng.binomial(1, prob_a1).astype(float)
        r = self.reward_fn(s, a) + rng.randn(n) * self.reward_noise_std

        marg_1 = self.pi_b_marg(s)
        pi_marg = np.where(a == 1, marg_1, 1 - marg_1)

        return {
            's': s,
            'a': a,
            'r': r,
            'u': u,
            'pi_b_marg': pi_marg,
        }

    def transition_coupling_diagnostic(self, n_samples=5000, seed=42):
        """
        Diagnostic: how much transition prediction depends on confounded vs clean dims.

        Returns MSE gap between using all dynamics dims vs clean-only dims.
        """
        from sklearn.linear_model import LinearRegression

        rng = np.random.RandomState(seed)
        data = self.sample_trajectories(n_samples, rng=rng)

        s_flat = data['s'].reshape(-1, self.state_dim)
        a_flat = data['a'].reshape(-1, 1)
        s_next_flat = data['s_next'].reshape(-1, self.state_dim)

        # Target: dynamics dims of next state
        target = s_next_flat[:, self.idx_dynamics]

        # All dynamics dims
        X_all = np.hstack([s_flat[:, self.idx_dynamics], a_flat])
        reg_all = LinearRegression().fit(X_all, target)
        mse_all = np.mean((reg_all.predict(X_all) - target) ** 2)

        # Clean dims only (exclude confounded)
        if self.dim_clean > 0:
            X_clean = np.hstack([s_flat[:, self.idx_clean], a_flat])
            reg_clean = LinearRegression().fit(X_clean, target)
            mse_clean = np.mean((reg_clean.predict(X_clean) - target) ** 2)
        else:
            mse_clean = mse_all

        return {
            'mse_all_dynamics': mse_all,
            'mse_clean_only': mse_clean,
            'mse_ratio': mse_clean / max(mse_all, 1e-10),
            'gap': mse_clean - mse_all,
        }

    def true_value(self, pi_e_fn=None, n_mc=50000, discount=1.0, seed=42):
        """Compute V(π_e) by Monte Carlo. Default π_e = always action 1."""
        rng = np.random.RandomState(seed)
        n = n_mc
        H = self.horizon
        d = self.state_dim

        u = rng.binomial(1, 0.5, size=n).astype(float)
        s_t = rng.randn(n, d) + np.outer(2 * u - 1, self.w_conf)

        total_reward = np.zeros(n)
        for t in range(H):
            a_t = np.ones(n) if pi_e_fn is None else pi_e_fn(s_t)
            r_mean = self.reward_fn(s_t, a_t)
            r_t = r_mean + rng.randn(n) * self.reward_noise_std
            total_reward += (discount ** t) * r_t
            if t < H - 1:
                s_t = self.transition(s_t, a_t, u, rng)

        return float(np.mean(total_reward))

    def one_step_always_one_value(self):
        """Exact one-step value for the deterministic always-action-1 policy."""
        if self.horizon != 1:
            raise ValueError(
                "one_step_always_one_value is defined only when horizon=1"
            )
        # The initial Gaussian noise and symmetric U shift both have mean zero.
        return float(self.r_bias + self.w_ra)

    def __repr__(self):
        return (f"UnifiedCMDP(S_∅={self.dim_noise}, S_a={self.dim_instrument}, "
                f"S_Δ={self.dim_confounded}, S_r={self.dim_clean}, "
                f"H={self.horizon}, w_u={self.w_u}, cs={self.conf_strength})")


def flatten_trajectories(data):
    """Flatten trajectory data into per-transition tuples."""
    n, H, d = data['s'].shape
    return {
        's': data['s'].reshape(n * H, d),
        'a': data['a'].reshape(n * H),
        'r': data['r'].reshape(n * H),
        's_next': data['s_next'].reshape(n * H, d),
        'u': np.repeat(data['u'], H),
        'pi_b_marg': data['pi_b_marg'].reshape(n * H),
    }
