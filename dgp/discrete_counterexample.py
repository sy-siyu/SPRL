"""
Discrete counterexample DGP for Theorem 1.

States S = {0, 1, 2} (s1, s2, s3), Actions A = {0, 1}, Confounder U = {0, 1}.
Compression: φ(s1) = φ(s2) = z1, φ(s3) = z3.

Demonstrates that Γ_Z > Γ_S: compression can amplify sensitivity.
"""

import numpy as np


class DiscreteCounterexampleDGP:
    """Stores the full joint P(S,U,A,R) for the Theorem 1 counterexample."""

    def __init__(self, p_su, pi_b, rewards, compression_map=None):
        """
        Parameters
        ----------
        p_su : array (n_states, n_u)
            Joint distribution P(S=s, U=u). Must sum to 1.
        pi_b : array (n_states, n_u)
            Behavioral policy π_b(A=1 | S=s, U=u).
        rewards : array (n_states, n_actions)
            Expected reward r(s, a).
        compression_map : dict, optional
            Maps state index -> latent index. Default: {0:0, 1:0, 2:1}.
        """
        self.p_su = np.array(p_su, dtype=float)
        self.pi_b = np.array(pi_b, dtype=float)
        self.rewards = np.array(rewards, dtype=float)

        self.n_states = self.p_su.shape[0]
        self.n_u = self.p_su.shape[1]
        self.n_actions = 2

        assert np.isclose(self.p_su.sum(), 1.0), f"P(S,U) sums to {self.p_su.sum()}"
        assert (self.pi_b >= 0).all() and (self.pi_b <= 1).all()

        if compression_map is None:
            self.compression_map = {0: 0, 1: 0, 2: 1}
        else:
            self.compression_map = compression_map

        latent_vals = set(self.compression_map.values())
        self.n_latent = len(latent_vals)
        self.latent_indices = sorted(latent_vals)

        self._compute_derived()

    def _compute_derived(self):
        """Compute marginals, latent policies, etc."""
        # P(S=s)
        self.p_s = self.p_su.sum(axis=1)
        # P(U=u)
        self.p_u = self.p_su.sum(axis=0)

        # P(U=u | S=s)
        self.p_u_given_s = self.p_su / self.p_s[:, None]

        # Marginal behavioral policy π_b(A=1 | S=s)
        self.pi_b_marg = (self.pi_b * self.p_u_given_s).sum(axis=1)

        # Full action probabilities π_b(A=a | S=s, U=u) for a in {0,1}
        # pi_b_full[s, u, a]
        self.pi_b_full = np.zeros((self.n_states, self.n_u, self.n_actions))
        self.pi_b_full[:, :, 1] = self.pi_b
        self.pi_b_full[:, :, 0] = 1.0 - self.pi_b

        # Latent space distributions
        # P(Z=z, U=u)
        self.p_zu = np.zeros((self.n_latent, self.n_u))
        for s, z in self.compression_map.items():
            z_idx = self.latent_indices.index(z)
            self.p_zu[z_idx] += self.p_su[s]

        # P(Z=z)
        self.p_z = self.p_zu.sum(axis=1)

        # P(S=s | Z=z, U=u)
        self.p_s_given_zu = np.zeros((self.n_states, self.n_latent, self.n_u))
        for s, z in self.compression_map.items():
            z_idx = self.latent_indices.index(z)
            for u in range(self.n_u):
                if self.p_zu[z_idx, u] > 0:
                    self.p_s_given_zu[s, z_idx, u] = self.p_su[s, u] / self.p_zu[z_idx, u]

        # Induced latent policy π_b(A=1 | Z=z, U=u)
        self.pi_b_z = np.zeros((self.n_latent, self.n_u))
        for z_idx in range(self.n_latent):
            for u in range(self.n_u):
                self.pi_b_z[z_idx, u] = np.sum(
                    self.p_s_given_zu[:, z_idx, u] * self.pi_b[:, u]
                )

        # Marginal latent policy π_b(A=1 | Z=z)
        self.p_u_given_z = self.p_zu / self.p_z[:, None]
        self.pi_b_z_marg = (self.pi_b_z * self.p_u_given_z).sum(axis=1)

        # Latent rewards r(z, a) = E[R | Z=z, A=a]
        self.rewards_z = np.zeros((self.n_latent, self.n_actions))
        for z_idx, z_val in enumerate(self.latent_indices):
            states_in_z = [s for s, z in self.compression_map.items() if z == z_val]
            for a in range(self.n_actions):
                # r(z,a) = E[R | Z=z, A=a] = Σ_s P(S=s|Z=z, A=a) r(s,a)
                # P(S=s|Z=z,A=a) = P(A=a|S=s,Z=z) P(S=s|Z=z) / P(A=a|Z=z)
                # = P(A=a|S=s) P(S=s|Z=z) / P(A=a|Z=z)  [since A⊥Z|S]
                # Simpler: use marginals
                p_s_given_z = np.zeros(self.n_states)
                for s in states_in_z:
                    p_s_given_z[s] = self.p_s[s] / self.p_z[z_idx]

                # P(S=s | Z=z, A=a) ∝ P(A=a|S=s) P(S=s|Z=z)
                # = π_b_marg(a|s) * P(s|z)
                if a == 1:
                    weights = self.pi_b_marg[states_in_z] * p_s_given_z[states_in_z]
                else:
                    weights = (1 - self.pi_b_marg[states_in_z]) * p_s_given_z[states_in_z]

                if weights.sum() > 0:
                    p_s_given_za = weights / weights.sum()
                    self.rewards_z[z_idx, a] = np.sum(
                        p_s_given_za * self.rewards[states_in_z, a]
                    )

        # Joint P(Z=z, A=a)
        self.p_za = np.zeros((self.n_latent, self.n_actions))
        for z_idx in range(self.n_latent):
            self.p_za[z_idx, 1] = self.p_z[z_idx] * self.pi_b_z_marg[z_idx]
            self.p_za[z_idx, 0] = self.p_z[z_idx] * (1 - self.pi_b_z_marg[z_idx])

    def compute_gamma_s(self):
        """Compute Γ_S: max odds ratio across all states."""
        gamma = 1.0
        for s in range(self.n_states):
            for u1 in range(self.n_u):
                for u2 in range(self.n_u):
                    p1 = self.pi_b[s, u1]
                    p2 = self.pi_b[s, u2]
                    p1 = np.clip(p1, 1e-12, 1 - 1e-12)
                    p2 = np.clip(p2, 1e-12, 1 - 1e-12)
                    odds1 = p1 / (1 - p1)
                    odds2 = p2 / (1 - p2)
                    ratio = odds1 / odds2
                    gamma = max(gamma, ratio)
        return gamma

    def compute_gamma_z(self):
        """Compute Γ_Z: max odds ratio across all latent states."""
        gamma = 1.0
        for z_idx in range(self.n_latent):
            for u1 in range(self.n_u):
                for u2 in range(self.n_u):
                    p1 = self.pi_b_z[z_idx, u1]
                    p2 = self.pi_b_z[z_idx, u2]
                    p1 = np.clip(p1, 1e-12, 1 - 1e-12)
                    p2 = np.clip(p2, 1e-12, 1 - 1e-12)
                    odds1 = p1 / (1 - p1)
                    odds2 = p2 / (1 - p2)
                    ratio = odds1 / odds2
                    gamma = max(gamma, ratio)
        return gamma

    def compute_delta(self):
        """Compute Δ = max_{z,u} ||P(S|Z=z,U=u) - P(S|Z=z)||_1."""
        delta = 0.0
        p_s_given_z = np.zeros((self.n_states, self.n_latent))
        for s, z in self.compression_map.items():
            z_idx = self.latent_indices.index(z)
            if self.p_z[z_idx] > 0:
                p_s_given_z[s, z_idx] = self.p_s[s] / self.p_z[z_idx]

        for z_idx in range(self.n_latent):
            for u in range(self.n_u):
                diff = np.abs(self.p_s_given_zu[:, z_idx, u] - p_s_given_z[:, z_idx])
                delta = max(delta, diff.sum())
        return delta

    def true_value(self, pi_e_s):
        """
        Compute V(π_e) = E_s[Σ_a π_e(a|s) r(s,a)] in the raw state space.

        Parameters
        ----------
        pi_e_s : array (n_states,)
            Target policy π_e(A=1|S=s).
        """
        v = 0.0
        for s in range(self.n_states):
            v += self.p_s[s] * (
                pi_e_s[s] * self.rewards[s, 1]
                + (1 - pi_e_s[s]) * self.rewards[s, 0]
            )
        return v

    def true_value_latent(self, pi_e_z):
        """
        Compute V(π_e) = E_z[Σ_a π_e(a|z) r(z,a)] in the latent space.

        Parameters
        ----------
        pi_e_z : array (n_latent,)
            Target policy π_e(A=1|Z=z).
        """
        v = 0.0
        for z_idx in range(self.n_latent):
            v += self.p_z[z_idx] * (
                pi_e_z[z_idx] * self.rewards_z[z_idx, 1]
                + (1 - pi_e_z[z_idx]) * self.rewards_z[z_idx, 0]
            )
        return v

    def sample(self, n, rng=None):
        """
        Sample (s, u, a, r) tuples from the DGP.

        Returns dict with arrays: 's', 'u', 'a', 'r', 'z'.
        """
        if rng is None:
            rng = np.random.default_rng()

        probs = self.p_su.ravel()
        su_flat = rng.choice(len(probs), size=n, p=probs)

        s = su_flat // self.n_u
        u = su_flat % self.n_u

        # Sample actions
        p_a1 = self.pi_b[s, u]
        a = (rng.random(n) < p_a1).astype(int)

        # Sample rewards (add small noise)
        r = self.rewards[s, a] + rng.normal(0, 0.01, size=n)

        # Compute z
        z = np.array([self.compression_map[si] for si in s])

        return {'s': s, 'u': u, 'a': a, 'r': r, 'z': z}

    def print_summary(self):
        """Print a summary of the DGP."""
        gamma_s = self.compute_gamma_s()
        gamma_z = self.compute_gamma_z()
        delta = self.compute_delta()

        print("=" * 60)
        print("Discrete Counterexample DGP Summary")
        print("=" * 60)

        print(f"\nJoint P(S,U):")
        for s in range(self.n_states):
            for u in range(self.n_u):
                print(f"  P(s{s+1}, u={u}) = {self.p_su[s, u]:.6f}")

        print(f"\nBehavioral policy π_b(A=1|S,U):")
        for s in range(self.n_states):
            for u in range(self.n_u):
                print(f"  π_b(1|s{s+1}, u={u}) = {self.pi_b[s, u]:.6f}")

        print(f"\nMarginal policy π_b(A=1|S):")
        for s in range(self.n_states):
            print(f"  π_b(1|s{s+1}) = {self.pi_b_marg[s]:.6f}")

        print(f"\nCompression: φ(s1)=φ(s2)=z1, φ(s3)=z3")

        print(f"\nP(S|Z=z1, U):")
        for s in range(self.n_states):
            for u in range(self.n_u):
                print(f"  P(s{s+1}|z1, u={u}) = {self.p_s_given_zu[s, 0, u]:.6f}")

        print(f"\nLatent policy π_b(A=1|Z,U):")
        for z_idx in range(self.n_latent):
            z_name = f"z{self.latent_indices[z_idx]+1}"
            for u in range(self.n_u):
                print(f"  π_b(1|{z_name}, u={u}) = {self.pi_b_z[z_idx, u]:.6f}")

        print(f"\nΓ_S = {gamma_s:.6f}")
        print(f"Γ_Z = {gamma_z:.6f}")
        print(f"Amplification = {gamma_z / gamma_s:.3f}x")
        print(f"Δ = {delta:.6f}")
        print(f"\nRewards r(s,a):")
        for s in range(self.n_states):
            for a in range(self.n_actions):
                print(f"  r(s{s+1}, a={a}) = {self.rewards[s, a]:.4f}")
        print(f"\nLatent rewards r(z,a):")
        for z_idx in range(self.n_latent):
            z_name = f"z{self.latent_indices[z_idx]+1}"
            for a in range(self.n_actions):
                print(f"  r({z_name}, a={a}) = {self.rewards_z[z_idx, a]:.4f}")


def build_counterexample(gamma_s_target=2.0, alpha=0.95, q=0.9, r=0.1):
    """
    Build a clean analytic counterexample for Theorem 1.

    The construction exploits two mechanisms:
    1. U determines which state dominates within z1:
       P(s1|z1,u=0) = α ≈ 1, P(s2|z1,u=1) = α ≈ 1
    2. Per-state policies have the same confounding direction:
       s1: high under u=0, s2: low under u=0
       But the per-state odds ratios are bounded by Γ_S.

    The latent odds ratio Γ_Z → odds(p)/odds(s) ≈ Γ_S² · odds(q)/odds(r)
    as α → 1, which can vastly exceed Γ_S.

    Parameters
    ----------
    gamma_s_target : float
        Target per-state sensitivity parameter Γ_S.
    alpha : float
        Mixing weight P(s1|z1,u=0) = P(s2|z1,u=1) = α.
        Higher α → more amplification.
    q : float
        π_b(1|s1,u=1). Higher q → more amplification.
    r : float
        π_b(1|s2,u=0). Lower r → more amplification.

    Returns
    -------
    DiscreteCounterexampleDGP
    """
    Gamma = gamma_s_target
    beta = 1 - alpha  # P(s1|z1,u=1) = β = 1-α

    # Derive p and s from the Γ_S constraint:
    # s1: odds(p)/odds(q) = Γ_S  →  p/(1-p) = Γ * q/(1-q)
    odds_q = q / (1 - q)
    odds_p = Gamma * odds_q
    p = odds_p / (1 + odds_p)

    # s2: odds(r)/odds(s) = Γ_S  →  s/(1-s) = odds(r)/Γ
    odds_r = r / (1 - r)
    odds_s = odds_r / Gamma
    s = odds_s / (1 + odds_s)

    # Verify per-state Γ
    assert abs((p / (1 - p)) / (q / (1 - q)) - Gamma) < 1e-10, \
        f"s1 Γ check failed: {(p/(1-p))/(q/(1-q))}"
    assert abs((r / (1 - r)) / (s / (1 - s)) - Gamma) < 1e-10, \
        f"s2 Γ check failed: {(r/(1-r))/(s/(1-s))}"

    # Construct P(S,U): U determines which state dominates in z1
    pz1_u0 = 0.4  # P(Z=z1, U=0)
    pz1_u1 = 0.4  # P(Z=z1, U=1)
    ps3_u0 = 0.1  # P(S=s3, U=0)
    ps3_u1 = 0.1  # P(S=s3, U=1)

    p_su = np.array([
        [alpha * pz1_u0, beta * pz1_u1],         # s1: dominant when u=0
        [(1-alpha) * pz1_u0, (1-beta) * pz1_u1], # s2: dominant when u=1
        [ps3_u0, ps3_u1],                          # s3: alone in z3
    ])
    assert np.isclose(p_su.sum(), 1.0)

    # Behavioral policy: s1 and s2 confounded in same direction
    # but with very different policy values
    pi_b = np.array([
        [p, q],     # s1: high under u=0 (odds ratio = Γ)
        [r, s],     # s2: low under u=0  (odds ratio = Γ)
        [0.5, 0.5], # s3: no confounding
    ])

    # Rewards: opposite for s1 and s2 (amplifies value distortion)
    rewards = np.array([
        [0.0, 1.0],  # s1: action 1 is good
        [1.0, 0.0],  # s2: action 0 is good
        [0.5, 0.5],  # s3: neutral
    ])

    dgp = DiscreteCounterexampleDGP(p_su, pi_b, rewards)

    return dgp


if __name__ == "__main__":
    print("Building analytic counterexample...")
    dgp = build_counterexample(gamma_s_target=2.0, alpha=0.95, q=0.9, r=0.1)
    dgp.print_summary()

    gamma_s = dgp.compute_gamma_s()
    gamma_z = dgp.compute_gamma_z()
    print(f"\nAmplification: Γ_Z/Γ_S = {gamma_z/gamma_s:.4f}x")
