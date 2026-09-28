"""Known-answer checks for the published loss and interval evaluator."""

import unittest

import numpy as np
import torch

from estimators.robust_ipw_continuous import robust_ipw_bounds_samplewise
from representations.sprl_cell import propensity_homogeneity_loss, SPRL, train_sprl
from dgp.discrete_counterexample import build_counterexample
from experiments.exact_one_step import sharp_policy_bounds
from experiments.exact_outcome_h2 import run as outcome_construction
from experiments.exact_transition_h2 import run as transition_construction


class IntervalTests(unittest.TestCase):
    def test_signed_reward_known_answer(self):
        # Exact observed distribution represented as six equally weighted rows:
        # U=0: one treated Y=2, two controls; U=1: two treated Y=-1, one control.
        a = np.array([1, 0, 0, 1, 1, 0])
        r = np.array([2, 0, 0, -1, -1, 0])
        low, high, _ = robust_ipw_bounds_samplewise(a, r, a, 2, np.full(6, .5))
        np.testing.assert_allclose([low, high], [-.5, .5], atol=1e-14)
        self.assertLessEqual(low, .5)
        self.assertGreaterEqual(high + 1e-14, .5)

    def test_unit_level_is_ipw(self):
        a = np.array([0, 1, 1, 0])
        r = np.array([-2., 3., -4., 5.])
        e = np.array([.2, .3, .6, .8])
        pi = np.array([.7, .3, .3, .7])
        expected = np.mean(pi * r / np.where(a == 1, e, 1-e))
        low, high, _ = robust_ipw_bounds_samplewise(a, r, pi, 1, e)
        np.testing.assert_allclose([low, high], [expected, expected])

    def test_width_identity_and_monotonicity(self):
        rng = np.random.RandomState(18)
        a = rng.binomial(1, .5, 100)
        r = rng.normal(size=100)
        e = rng.uniform(.05, .95, 100)
        old = None
        for level in [1., 1.4, 2., 3.]:
            low, high, _ = robust_ipw_bounds_samplewise(a, r, a, level, e)
            width = (level - 1 / level) * np.mean(a * abs(r) * (1-e)/e)
            self.assertAlmostEqual(high-low, width)
            if old is not None:
                self.assertLessEqual(low, old[0] + 1e-12)
                self.assertGreaterEqual(high, old[1] - 1e-12)
            old = low, high

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            robust_ipw_bounds_samplewise([1], [2], [1], .5, [.5])
        with self.assertRaises(ValueError):
            robust_ipw_bounds_samplewise([2], [2], [1], 2, [.5])


class LossTests(unittest.TestCase):
    def test_pair_loss_and_gradient(self):
        z = torch.tensor([[0.], [1.], [3.]], requires_grad=True)
        e = torch.tensor([.1, .4, .9])
        loss = propensity_homogeneity_loss(z, e, sigma=1.)
        w = torch.exp(-torch.cdist(z, z).square()/2.)
        w = w * (1-torch.eye(3))
        expected = (w * (e[:,None]-e[None,:]).square()).sum()/(w.sum()+1e-8)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(float(z.grad.abs().sum()), 0.)

    def test_constant_teacher_has_zero_loss_and_gradient(self):
        z = torch.randn(5, 2, requires_grad=True)
        loss = propensity_homogeneity_loss(z, torch.full((5,), .5))
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.)
        self.assertEqual(float(z.grad.abs().sum()), 0.)

    def test_smoke_training_activates_propensity_term(self):
        torch.set_num_threads(1)
        torch.manual_seed(10)
        rng = np.random.RandomState(10)
        s = rng.normal(size=(32, 3))
        a = rng.binomial(1, .5, size=32)
        r = s[:,0] + .2*a
        e = 1/(1+np.exp(-s[:,1]))
        model = SPRL(3, 2, use_transitions=False)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        losses = train_sprl(model, s, a, r, e, lambda_prop=1., epochs=2,
                            batch_size=16, warmup_epochs=0, lambda_ramp_epochs=0,
                            device="cpu", verbose=False)
        self.assertGreater(losses[-1]["prop"], 0.)
        self.assertTrue(any(not torch.equal(before[k], v)
                            for k, v in model.state_dict().items()))


class FiniteConstructionTests(unittest.TestCase):
    def test_one_step_sharp_bounds(self):
        dgp = build_counterexample(gamma_s_target=2., alpha=.95, q=.9, r=.1)
        self.assertAlmostEqual(dgp.compute_gamma_s(), 2.)
        np.testing.assert_allclose(sharp_policy_bounds(dgp, "raw", np.ones(3), 2.), [.5,.5])
        low, high = sharp_policy_bounds(dgp, "latent", np.ones(2), 2.)
        self.assertEqual([round(low,3),round(high,3)], [.834,.867])

    def test_sequential_constructions(self):
        # Each run also checks normalization and the construction-specific
        # inclusion/exclusion statements before returning.
        self.assertIsInstance(outcome_construction(horizon=2), dict)
        self.assertIsInstance(transition_construction(), dict)


if __name__ == "__main__":
    unittest.main()
