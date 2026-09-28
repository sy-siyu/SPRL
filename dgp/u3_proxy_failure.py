"""Exact three-category-confounder stress test for the propensity proxy.

The construction has two independent observed components:

``T``
    A task variable.  The structural reward is
    ``2 * T - 1 + action_reward_effect * A``.
``C``
    A confounded alias variable.  The hidden variable ``U`` changes the
    distribution of ``C`` and the behavioral policy depends on ``(C, U)``.

The observed state is the concatenation of one-hot encodings of ``T`` and
``C``.  The marginal action-1 propensity is exactly one half at every raw
state.  Consequently, a representation that keeps ``T`` and discards ``C``
has zero population propensity-homogeneity loss, even though its pairwise
sensitivity is larger than the raw-state sensitivity.

This module is deliberately separate from :mod:`dgp.unified_cmdp`: it is a
focused, exactly enumerable stress test and does not alter the binary-U
experimental pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from typing import Any

import numpy as np


def _odds(probability: np.ndarray | float) -> np.ndarray:
    probability_array = np.asarray(probability, dtype=float)
    return probability_array / (1.0 - probability_array)


class U3ProxyFailureDGP:
    """Moderate exact-homogeneity counterexample with ``|U| = 3``.

    Parameters
    ----------
    u_labels:
        Three distinct labels returned in ``sample()["u"]``.  The numerical
        construction uses stable internal indices, so relabeling categories
        cannot change any population quantity.
    reward_noise_std:
        Default standard deviation of optional mean-zero Gaussian reward
        noise.
    action_reward_effect:
        Direct effect of the binary action on reward.  This makes the
        controlled stress test value-relevant while leaving the propensity
        construction and its exact sensitivity values unchanged.

    Notes
    -----
    Prototype-state order is ``(T, C) = (0, 0), (0, 1), (1, 0), (1, 1)``.
    Thus ``identity_mapping`` retains both state components and
    ``drop_c_mapping`` maps the four states to ``(0, 0, 1, 1)``, retaining
    only ``T``.
    """

    # Audited moderate construction: minimum P(U=u|C=c)=.10 and all
    # behavioral probabilities lie in [.38, .62].
    _MINIMUM_CELL = 0.10
    _POLICY_FLOOR = 0.38
    _MIDDLE_MASS = 0.2795604968479981

    def __init__(
        self,
        u_labels: Sequence[Hashable] = (0, 1, 2),
        reward_noise_std: float = 0.0,
        action_reward_effect: float = 0.25,
    ) -> None:
        labels = tuple(u_labels)
        if len(labels) != 3 or len(set(labels)) != 3:
            raise ValueError("u_labels must contain exactly three distinct labels.")
        if reward_noise_std < 0:
            raise ValueError("reward_noise_std must be non-negative.")
        if not np.isfinite(action_reward_effect):
            raise ValueError("action_reward_effect must be finite.")

        self.u_labels = labels
        self.reward_noise_std = float(reward_noise_std)
        self.action_reward_effect = float(action_reward_effect)
        self.state_dim = 4
        self.n_u = 3
        self.n_actions = 2

        minimum_cell = self._MINIMUM_CELL
        middle_mass = self._MIDDLE_MASS
        remaining_mass = 1.0 - minimum_cell - middle_mass
        eta = self._POLICY_FLOOR
        centered_probability = (
            0.5 - middle_mass * (1.0 - eta)
        ) / (1.0 - middle_mass)

        self._p_u_given_c = np.array(
            [
                [minimum_cell, middle_mass, remaining_mass],
                [middle_mass, minimum_cell, remaining_mass],
            ],
            dtype=float,
        )
        self._policy = np.array(
            [
                [centered_probability, 1.0 - eta, centered_probability],
                [eta, 1.0 - centered_probability, 1.0 - centered_probability],
            ],
            dtype=float,
        )

        # The supplied table is P(U|C), with P(C)=1/2.  Sampling is written in
        # causal order U -> C by applying Bayes' rule once here.
        self._p_c = np.full(2, 0.5)
        self._p_t = np.full(2, 0.5)
        self._p_u = self._p_c @ self._p_u_given_c
        self._p_c_given_u = (
            self._p_c[:, None] * self._p_u_given_c / self._p_u[None, :]
        )
        self._marginal_propensity = np.sum(
            self._p_u_given_c * self._policy,
            axis=1,
        )

        self.identity_mapping = np.arange(4, dtype=int)
        self.drop_c_mapping = np.array([0, 0, 1, 1], dtype=int)
        self.keep_c_mapping = np.array([0, 1, 0, 1], dtype=int)
        self.full_merge_mapping = np.zeros(4, dtype=int)

        self._validate_construction()

    @property
    def p_u_given_c(self) -> np.ndarray:
        """Return the exact table ``P(U=u | C=c)`` with shape ``(2, 3)``."""
        return self._p_u_given_c.copy()

    @property
    def policy(self) -> np.ndarray:
        """Return ``P(A=1 | C=c, U=u)`` with shape ``(2, 3)``."""
        return self._policy.copy()

    @property
    def p_u(self) -> np.ndarray:
        """Return the marginal distribution of the three hidden categories."""
        return self._p_u.copy()

    @property
    def p_c_given_u(self) -> np.ndarray:
        """Return ``P(C=c | U=u)`` with shape ``(2, 3)``."""
        return self._p_c_given_u.copy()

    @property
    def marginal_propensity(self) -> np.ndarray:
        """Return ``P(A=1 | C=c)``; both entries equal exactly ``0.5``."""
        return self._marginal_propensity.copy()

    @property
    def minimum_hidden_cell_mass(self) -> float:
        return float(self._p_u_given_c.min())

    @property
    def minimum_policy_probability(self) -> float:
        """Smallest probability over both binary actions."""
        return float(np.minimum(self._policy, 1.0 - self._policy).min())

    @property
    def maximum_action1_probability(self) -> float:
        return float(self._policy.max())

    @property
    def maximum_policy_probability(self) -> float:
        """Largest probability over both binary actions."""
        return float(np.maximum(self._policy, 1.0 - self._policy).max())

    @property
    def gamma_s(self) -> float:
        """Exact pairwise raw-state sensitivity."""
        return self.gamma_for_mapping(self.identity_mapping)

    @property
    def gamma_merged(self) -> float:
        """Exact sensitivity after retaining ``T`` and discarding ``C``."""
        return self.gamma_for_mapping(self.drop_c_mapping)

    @property
    def amplification_merged(self) -> float:
        return self.gamma_merged / self.gamma_s

    def prototype_states(self, return_labels: bool = False):
        """Return the four concatenated one-hot ``(T, C)`` state prototypes.

        If ``return_labels`` is true, return ``(states, t, c)``.
        """
        t = np.array([0, 0, 1, 1], dtype=int)
        c = np.array([0, 1, 0, 1], dtype=int)
        states = self.encode_state(t, c)
        if return_labels:
            return states, t, c
        return states

    @staticmethod
    def encode_state(t: np.ndarray | int, c: np.ndarray | int) -> np.ndarray:
        """Concatenate binary one-hot encodings of ``T`` and ``C``."""
        t_array, c_array = np.broadcast_arrays(
            np.asarray(t, dtype=int),
            np.asarray(c, dtype=int),
        )
        if not np.isin(t_array, (0, 1)).all() or not np.isin(c_array, (0, 1)).all():
            raise ValueError("T and C must be binary.")
        flat_t = t_array.ravel()
        flat_c = c_array.ravel()
        state = np.zeros((flat_t.size, 4), dtype=float)
        state[np.arange(flat_t.size), flat_t] = 1.0
        state[np.arange(flat_c.size), 2 + flat_c] = 1.0
        return state

    @staticmethod
    def decode_state(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decode concatenated one-hot states into ``(T, C)``."""
        state_array = np.atleast_2d(np.asarray(state, dtype=float))
        if state_array.shape[1] != 4:
            raise ValueError("state must have four columns: one-hot(T), one-hot(C).")
        t = np.argmax(state_array[:, :2], axis=1)
        c = np.argmax(state_array[:, 2:], axis=1)
        return t, c

    def behavior_probability(
        self,
        c: np.ndarray | int,
        u: np.ndarray | Hashable,
        *,
        u_is_index: bool = False,
    ) -> np.ndarray:
        """Evaluate ``P(A=1 | C, U)``.

        ``u`` is interpreted as public category labels unless ``u_is_index``
        is true.  The explicit flag avoids ambiguity when labels themselves
        are a permutation of ``(0, 1, 2)``.
        """
        c_array = np.asarray(c, dtype=int)
        u_array = np.asarray(u)
        c_array, u_array = np.broadcast_arrays(c_array, u_array)
        if not np.isin(c_array, (0, 1)).all():
            raise ValueError("C must be binary.")

        if u_is_index:
            u_index = np.asarray(u_array, dtype=int)
            if not np.isin(u_index, (0, 1, 2)).all():
                raise ValueError("Internal hidden-category indices must be 0, 1, or 2.")
        else:
            label_to_index = {
                label: index for index, label in enumerate(self.u_labels)
            }
            flat_indices = []
            for value in u_array.ravel():
                value_item = value.item() if hasattr(value, "item") else value
                if value_item not in label_to_index:
                    raise ValueError(f"Unknown hidden category {value_item!r}.")
                flat_indices.append(label_to_index[value_item])
            u_index = np.asarray(flat_indices, dtype=int).reshape(u_array.shape)
        return self._policy[c_array, u_index]

    def pi_b(self, state: np.ndarray, u: np.ndarray | Hashable) -> np.ndarray:
        """Evaluate ``P(A=1 | S, U)`` from encoded states."""
        _, c = self.decode_state(state)
        return self.behavior_probability(c, u).ravel()

    def pi_b_marg(self, state: np.ndarray) -> np.ndarray:
        """Evaluate the action-1 marginal propensity, exactly ``0.5``."""
        _, c = self.decode_state(state)
        return self._marginal_propensity[c]

    def _mapping_array(
        self,
        mapping: Sequence[Hashable] | Callable[[int, int], Hashable],
    ) -> np.ndarray:
        _, prototype_t, prototype_c = self.prototype_states(return_labels=True)
        if callable(mapping):
            mapped_values = [
                mapping(int(t), int(c))
                for t, c in zip(prototype_t, prototype_c)
            ]
        else:
            mapped_values = list(mapping)
            if len(mapped_values) != 4:
                raise ValueError("mapping must provide one label for each of four states.")

        mapped = np.empty(4, dtype=object)
        mapped[:] = mapped_values
        for value in mapped:
            try:
                hash(value.item() if hasattr(value, "item") else value)
            except TypeError as error:
                raise ValueError("mapping labels must be hashable.") from error
        return mapped

    def policy_for_mapping(
        self,
        mapping: Sequence[Hashable] | Callable[[int, int], Hashable],
    ) -> dict[str, Any]:
        """Exactly enumerate induced policies for an arbitrary state mapping.

        Returns a dictionary containing the cell labels, ``P(Z,U)``, and
        ``P(A=1|Z,U)``.  No Monte Carlo approximation is used.
        """
        mapped = self._mapping_array(mapping)
        _, prototype_t, prototype_c = self.prototype_states(return_labels=True)
        joint_state_u = (
            self._p_t[prototype_t, None]
            * self._p_c[prototype_c, None]
            * self._p_u_given_c[prototype_c]
        )

        cell_labels = []
        for value in mapped:
            value_item = value.item() if hasattr(value, "item") else value
            if value_item not in cell_labels:
                cell_labels.append(value_item)

        p_zu = np.zeros((len(cell_labels), self.n_u), dtype=float)
        numerator = np.zeros_like(p_zu)
        for state_index, cell_label in enumerate(mapped):
            label_item = (
                cell_label.item() if hasattr(cell_label, "item") else cell_label
            )
            cell_index = cell_labels.index(label_item)
            p_zu[cell_index] += joint_state_u[state_index]
            numerator[cell_index] += (
                joint_state_u[state_index]
                * self._policy[prototype_c[state_index]]
            )

        latent_policy = numerator / p_zu
        return {
            "cell_labels": tuple(cell_labels),
            "p_zu": p_zu,
            "policy": latent_policy,
            "mapping": mapped.copy(),
        }

    def gamma_for_mapping(
        self,
        mapping: Sequence[Hashable] | Callable[[int, int], Hashable],
    ) -> float:
        """Compute the exact maximum pairwise odds range after a mapping."""
        induced = self.policy_for_mapping(mapping)["policy"]
        induced_odds = _odds(induced)
        gamma_by_cell = induced_odds.max(axis=1) / induced_odds.min(axis=1)
        return float(gamma_by_cell.max())

    def enumerate_population(self) -> dict[str, np.ndarray]:
        """Enumerate all ``(T,C,U,A)`` atoms and their exact probabilities."""
        records: dict[str, list[Any]] = {
            "t": [],
            "c": [],
            "u_index": [],
            "u": [],
            "a": [],
            "s": [],
            "z_merged": [],
            "probability": [],
            "pi_b": [],
            "e": [],
            "r_mean": [],
        }
        for t in (0, 1):
            for c in (0, 1):
                state = self.encode_state(t, c)[0]
                merged = np.eye(2, dtype=float)[t]
                for u_index, u_label in enumerate(self.u_labels):
                    p_tcu = (
                        self._p_t[t]
                        * self._p_c[c]
                        * self._p_u_given_c[c, u_index]
                    )
                    p_a1 = self._policy[c, u_index]
                    for action in (0, 1):
                        records["t"].append(t)
                        records["c"].append(c)
                        records["u_index"].append(u_index)
                        records["u"].append(u_label)
                        records["a"].append(action)
                        records["s"].append(state)
                        records["z_merged"].append(merged)
                        records["probability"].append(
                            p_tcu * (p_a1 if action == 1 else 1.0 - p_a1)
                        )
                        records["pi_b"].append(p_a1)
                        records["e"].append(self._marginal_propensity[c])
                        records["r_mean"].append(
                            2.0 * t
                            - 1.0
                            + self.action_reward_effect * action
                        )

        result: dict[str, np.ndarray] = {}
        for key, values in records.items():
            if key == "u":
                result[key] = np.asarray(values)
            elif key in {"s", "z_merged"}:
                result[key] = np.asarray(values, dtype=float)
            elif key in {"t", "c", "u_index", "a"}:
                result[key] = np.asarray(values, dtype=int)
            else:
                result[key] = np.asarray(values, dtype=float)
        return result

    def audit_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable population audit of the construction."""
        merged = self.policy_for_mapping(self.drop_c_mapping)
        return {
            "u_labels": list(self.u_labels),
            "p_u_given_c": self._p_u_given_c.tolist(),
            "p_u": self._p_u.tolist(),
            "p_c_given_u": self._p_c_given_u.tolist(),
            "policy": self._policy.tolist(),
            "marginal_propensity": self._marginal_propensity.tolist(),
            "merged_policy": merged["policy"].tolist(),
            "minimum_hidden_cell_mass": self.minimum_hidden_cell_mass,
            "minimum_policy_probability": self.minimum_policy_probability,
            "maximum_policy_probability": self.maximum_policy_probability,
            "gamma_s": self.gamma_s,
            "gamma_merged": self.gamma_merged,
            "amplification_merged": self.amplification_merged,
            "action_reward_effect": self.action_reward_effect,
        }

    def sample(
        self,
        n: int,
        rng: np.random.Generator | np.random.RandomState | None = None,
        reward_noise_std: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Sample observations in causal order ``U -> C`` with independent T."""
        if n <= 0:
            raise ValueError("n must be positive.")
        if rng is None:
            rng = np.random.default_rng()
        noise_std = (
            self.reward_noise_std
            if reward_noise_std is None
            else float(reward_noise_std)
        )
        if noise_std < 0:
            raise ValueError("reward_noise_std must be non-negative.")

        u_index = rng.choice(self.n_u, size=n, p=self._p_u).astype(int)
        p_c1 = self._p_c_given_u[1, u_index]
        c = rng.binomial(1, p_c1, size=n).astype(int)
        t = rng.binomial(1, 0.5, size=n).astype(int)
        state = self.encode_state(t, c)

        p_a1 = self._policy[c, u_index]
        action = rng.binomial(1, p_a1, size=n).astype(int)
        reward_mean = (
            2.0 * t - 1.0 + self.action_reward_effect * action
        )
        reward = reward_mean + rng.normal(0.0, noise_std, size=n)
        e_action1 = self._marginal_propensity[c]
        observed_action_propensity = np.where(
            action == 1,
            e_action1,
            1.0 - e_action1,
        )
        u = np.asarray([self.u_labels[index] for index in u_index])

        return {
            "s": state,
            "t": t,
            "c": c,
            "u": u,
            "u_index": u_index,
            "a": action,
            "r": reward,
            "r_mean": reward_mean,
            "pi_b": p_a1,
            "e": e_action1,
            "pi_b_marg": observed_action_propensity,
            "z_merged": np.eye(2, dtype=float)[t],
        }

    def _validate_construction(self) -> None:
        if not np.allclose(self._p_u_given_c.sum(axis=1), 1.0, atol=1e-12):
            raise RuntimeError("Each P(U|C) row must sum to one.")
        if not np.allclose(self._p_c_given_u.sum(axis=0), 1.0, atol=1e-12):
            raise RuntimeError("Each P(C|U) column must sum to one.")
        if not np.allclose(self._marginal_propensity, 0.5, atol=1e-12):
            raise RuntimeError("The exact marginal propensity must equal one half.")
        if self.minimum_hidden_cell_mass < self._MINIMUM_CELL - 1e-12:
            raise RuntimeError("Hidden-cell mass fell below the audited minimum.")
        if self.minimum_policy_probability < self._POLICY_FLOOR - 1e-12:
            raise RuntimeError("Behavioral policy fell below the audited positivity floor.")


__all__ = ["U3ProxyFailureDGP"]
