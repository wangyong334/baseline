"""Policy targets must not use hindsight to choose actions."""
import unittest
import numpy as np
from tools.bst.data import Recording
from tools.bst.policy import BirthContext, policy_returns
from tools.bst.pipeline import threshold_costs


class PolicyTests(unittest.TestCase):
    def test_frozen_actions_not_oracle_minimum(self):
        z = np.array([[-2, 2], [-2, 2]], np.float32)
        y = np.array([0, 1], np.float32)
        valid = np.ones_like(z, bool)
        # Both events have identical observations; the fixed policy waits for
        # both, even though this is harmful for the first and helpful for second.
        target, ret, eligible = policy_returns(z, y, valid, valid, 0, 1, 1, 0.1)
        np.testing.assert_allclose(ret[:, 0], [1.1, 0.1])
        np.testing.assert_allclose(target[:, 0], [1, 0])
        self.assertTrue(eligible[:, 0].all())
        self.assertFalse(eligible[:, 1].any())

    def test_truncation_and_stop_action(self):
        z = np.array([[2, 2, -2], [-2, -2, 2]], np.float32)
        valid = np.array([[1, 1, 0], [1, 1, 1]], bool)
        waits = np.zeros_like(valid)
        target, ret, eligible = policy_returns(z, np.ones(2), valid, waits, 0, 1, 1, .1)
        np.testing.assert_allclose(ret[:, 0], [0, 1])
        np.testing.assert_allclose(target[:, 1], [0, 0])
        self.assertFalse(eligible[0, 1])

    def test_report_cost_mapping(self):
        tau, fp, fn = threshold_costs(.9, np.log(30))
        self.assertAlmostEqual(tau, np.log(fp / fn))
        self.assertLess(tau, 0)
        self.assertAlmostEqual(1 / (1 + np.exp(-(tau + np.log(30)))), .9)

    def test_birth_prefix_and_labels_do_not_leak(self):
        def rec(z, labels):
            return Recording("x", np.array([3, 4, 3]), np.array([3, 4, 3]),
                             np.array([1, 20, 70]), np.array([1, -1, 1]),
                             np.array(z, np.float32), np.array(labels, np.float32), 50)
        a = BirthContext(rec([-2, 1, 3], [0, 1, 1]), retain_all=True)
        b = BirthContext(rec([-2, 1, -100], [1, 0, 0]), retain_all=True)
        np.testing.assert_allclose(a.features([0, 1], 0), b.features([0, 1], 0))
        self.assertEqual(a.current, 0)
        a.features([2], 1)
        np.testing.assert_allclose(a.features([0, 1], 0), b.features([0, 1], 0))


if __name__ == "__main__":
    unittest.main()
