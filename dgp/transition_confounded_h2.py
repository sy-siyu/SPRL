"""Exact finite H=2 DGP with hidden transition confounding.

The construction is intentionally separate from the submitted Setting-B
generator and from ``run_multistep_coverage_exact.py``.  It is small enough to
enumerate exactly and is used to study transport of a transition-kernel
sensitivity level under state compression.

At time zero the raw state is S0 in {0, 1}, the hidden variable U in {0, 1}
is fixed for the trajectory, and the behavior chooses A0.  The transition
draws a binary outcome-state Y with a kernel that depends on U even after
conditioning on (S0, A0).  The second raw state is S1=(S0,Y), the behavior
chooses A1 independently with probability 1/2, and R=A1*Y.  The target policy
chooses action one at both times.  Compression drops S0 but retains Y at time
one, so it never merges different structural reward functions.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from typing import Sequence

import numpy as np


def _fraction(value: int | float | Fraction) -> Fraction:
    if isinstance(value, Fraction):
        return value
    return Fraction(str(value))


def _fraction_matrix(
    values: Sequence[Sequence[int | float | Fraction]],
) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(tuple(_fraction(value) for value in row) for row in values)


def _fraction_tensor(
    values: Sequence[Sequence[Sequence[int | float | Fraction]]],
) -> tuple[tuple[tuple[Fraction, ...], ...], ...]:
    return tuple(_fraction_matrix(matrix) for matrix in values)


class TransitionConfoundedH2DGP:
    """Stores the exact population law for the transition-confounded example."""

    def __init__(
        self,
        p_su: Sequence[Sequence[int | float | Fraction]] | None = None,
        pi0: Sequence[Sequence[int | float | Fraction]] | None = None,
        transition_y1: (
            Sequence[Sequence[Sequence[int | float | Fraction]]] | None
        ) = None,
        pi1: int | float | Fraction = Fraction(1, 2),
    ) -> None:
        # Rows index S0 and columns index U.
        self.p_su_q = _fraction_matrix(
            p_su
            if p_su is not None
            else (
                (Fraction(2, 5), Fraction(3, 10)),
                (Fraction(1, 10), Fraction(1, 5)),
            )
        )
        self.pi0_q = _fraction_matrix(
            pi0
            if pi0 is not None
            else (
                (Fraction(1, 5), Fraction(1, 5)),
                (Fraction(3, 5), Fraction(4, 5)),
            )
        )
        # First index is A0, followed by S0 and U.  The A0=0 row completes
        # the MDP with overlap; the target value uses the A0=1 row.
        self.transition_y1_q = _fraction_tensor(
            transition_y1
            if transition_y1 is not None
            else (
                (
                    (Fraction(1, 2), Fraction(1, 2)),
                    (Fraction(1, 2), Fraction(1, 2)),
                ),
                (
                    (Fraction(1, 10), Fraction(1, 10)),
                    (Fraction(1, 10), Fraction(9, 10)),
                ),
            )
        )
        self.pi1_q = _fraction(pi1)

        self.n_states = 2
        self.n_u = 2
        self.n_actions = 2
        self._validate()

        self.p_su = np.asarray(
            [[float(value) for value in row] for row in self.p_su_q],
            dtype=float,
        )
        self.pi0 = np.asarray(
            [[float(value) for value in row] for row in self.pi0_q],
            dtype=float,
        )
        self.transition_y1 = np.asarray(
            [
                [[float(value) for value in row] for row in matrix]
                for matrix in self.transition_y1_q
            ],
            dtype=float,
        )
        self.pi1 = float(self.pi1_q)

    def _validate(self) -> None:
        if len(self.p_su_q) != 2 or any(len(row) != 2 for row in self.p_su_q):
            raise ValueError("p_su must have shape (2, 2)")
        if sum(sum(row) for row in self.p_su_q) != 1:
            raise ValueError("p_su must sum exactly to one")
        if len(self.pi0_q) != 2 or any(len(row) != 2 for row in self.pi0_q):
            raise ValueError("pi0 must have shape (2, 2)")
        if len(self.transition_y1_q) != 2:
            raise ValueError("transition_y1 must have shape (2, 2, 2)")
        if any(
            len(matrix) != 2 or any(len(row) != 2 for row in matrix)
            for matrix in self.transition_y1_q
        ):
            raise ValueError("transition_y1 must have shape (2, 2, 2)")

        probabilities = [
            *[value for row in self.p_su_q for value in row],
            *[value for row in self.pi0_q for value in row],
            *[
                value
                for matrix in self.transition_y1_q
                for row in matrix
                for value in row
            ],
            self.pi1_q,
        ]
        if any(value < 0 or value > 1 for value in probabilities):
            raise ValueError("All entries must be probabilities")
        if any(value <= 0 for row in self.p_su_q for value in row):
            raise ValueError("Every (S0,U) cell must have positive mass")
        if any(value in (0, 1) for row in self.pi0_q for value in row):
            raise ValueError("The time-zero behavior policy must have overlap")
        if self.pi1_q in (0, 1):
            raise ValueError("The time-one behavior policy must have overlap")
        if any(
            value in (0, 1)
            for matrix in self.transition_y1_q
            for row in matrix
            for value in row
        ):
            raise ValueError("Every transition outcome must have positive support")

    def state_mass_q(self, state: int) -> Fraction:
        return sum(self.p_su_q[state])

    def hidden_mass_q(self, u: int) -> Fraction:
        return sum(self.p_su_q[state][u] for state in range(self.n_states))

    def behavior_action_one_given_state_q(self, state: int) -> Fraction:
        numerator = sum(
            self.p_su_q[state][u] * self.pi0_q[state][u]
            for u in range(self.n_u)
        )
        return numerator / self.state_mass_q(state)

    def causal_transition_y1_given_state_q(
        self, state: int, action: int = 1
    ) -> Fraction:
        """P(Y=1 | S0=s, do(A0=a))."""
        numerator = sum(
            self.p_su_q[state][u] * self.transition_y1_q[action][state][u]
            for u in range(self.n_u)
        )
        return numerator / self.state_mass_q(state)

    def observed_transition_y1_given_state_q(
        self, state: int, action: int = 1
    ) -> Fraction:
        """P(Y=1 | S0=s, A0=a) under the behavior law."""
        numerator = Fraction(0)
        denominator = Fraction(0)
        for u in range(self.n_u):
            action_probability = (
                self.pi0_q[state][u]
                if action == 1
                else 1 - self.pi0_q[state][u]
            )
            mass = self.p_su_q[state][u] * action_probability
            denominator += mass
            numerator += mass * self.transition_y1_q[action][state][u]
        return numerator / denominator

    def causal_transition_y1_latent_q(self, action: int = 1) -> Fraction:
        return sum(
            self.p_su_q[state][u]
            * self.transition_y1_q[action][state][u]
            for state in range(self.n_states)
            for u in range(self.n_u)
        )

    def observed_transition_y1_latent_q(self, action: int = 1) -> Fraction:
        numerator = Fraction(0)
        denominator = Fraction(0)
        for state in range(self.n_states):
            for u in range(self.n_u):
                action_probability = (
                    self.pi0_q[state][u]
                    if action == 1
                    else 1 - self.pi0_q[state][u]
                )
                mass = self.p_su_q[state][u] * action_probability
                denominator += mass
                numerator += mass * self.transition_y1_q[action][state][u]
        return numerator / denominator

    def latent_action_one_given_u_q(self, u: int) -> Fraction:
        numerator = sum(
            self.p_su_q[state][u] * self.pi0_q[state][u]
            for state in range(self.n_states)
        )
        return numerator / self.hidden_mass_q(u)

    def target_value_q(self) -> Fraction:
        """Value of the target A0=A1=1."""
        return self.causal_transition_y1_latent_q(action=1)

    @staticmethod
    def reward(state0: int, u: int, action1: int, y: int) -> int:
        """Structural reward; state0 and U are deliberately irrelevant."""
        del state0, u
        return action1 * y

    def enumerate_behavior_trajectories(self) -> list[dict]:
        """Enumerate P(S0,U,A0,Y,A1,R) exactly."""
        records: list[dict] = []
        for state, u, action0, y, action1 in product(
            range(2), range(2), range(2), range(2), range(2)
        ):
            p_action0 = (
                self.pi0_q[state][u]
                if action0 == 1
                else 1 - self.pi0_q[state][u]
            )
            p_y1 = self.transition_y1_q[action0][state][u]
            p_y = p_y1 if y == 1 else 1 - p_y1
            p_action1 = self.pi1_q if action1 == 1 else 1 - self.pi1_q
            probability = (
                self.p_su_q[state][u] * p_action0 * p_y * p_action1
            )
            records.append(
                {
                    "state0": state,
                    "u": u,
                    "action0": action0,
                    "y": y,
                    "state1": (state, y),
                    "latent0": 0,
                    "latent1": y,
                    "action1": action1,
                    "reward": self.reward(state, u, action1, y),
                    "probability": probability,
                }
            )
        if sum(record["probability"] for record in records) != 1:
            raise AssertionError("Behavior trajectory probabilities must sum to one")
        return records
