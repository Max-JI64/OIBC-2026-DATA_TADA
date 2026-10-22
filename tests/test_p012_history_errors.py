"""Allocation and preregistered evidence contracts; no data or model fit."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from p012_history_errors import allocate, evidence


def row():
    return dict(short=dict(sites=10, mean_delta_cf_mae=.01, median_delta_cf_mae=.001, worsening_share=.6),
                longer=dict(sites=20, mean_delta_cf_mae=-.01), short_allocated_delta_W=10.)


class HistoryContracts(unittest.TestCase):
    def test_efficiency_with_zero_and_sign_crossing(self):
        b1, b2, p1, p2 = [np.array(x, float) for x in ([2, -3, 0], [-1, 1, 0], [-4, 2, 1], [3, 1, -1])]
        q = np.array([3., 1., 0.])
        r = allocate(b1, b2, p1, p2, q)
        np.testing.assert_allclose(r["short"]+r["longer"], r["after"]-r["before"])
        self.assertEqual(r["short"][-1], 0)

    def test_symmetry_and_unchanged_group(self):
        a = np.array([2., -4.]); b = np.array([3., 2.]); p = np.array([-1., 5.]); q = np.ones(2)
        r = allocate(a, b, p, b, q)
        np.testing.assert_allclose(r["longer"], 0)
        swapped = allocate(b, a, b, p, q)
        np.testing.assert_allclose(r["short"], swapped["longer"])

    def test_no_internal_subset_W_additivity(self):
        r = allocate(np.array([10.]), np.array([-10.]), np.array([5.]), np.array([-4.]), np.ones(1))
        self.assertEqual(r["before"][0], 0.)
        self.assertEqual(r["after"][0], 1.)
        self.assertAlmostEqual((r["short"]+r["longer"])[0], 1.)

    def test_bad_weights_rejected(self):
        with self.assertRaises(ValueError):
            allocate(np.ones(1), np.ones(1), np.ones(1), np.ones(1), -np.ones(1))

    def test_all_conditions_required(self):
        r = row()
        self.assertTrue(evidence(r)["passes"])
        for field, value in (("sites", 9), ("median_delta_cf_mae", 0.), ("worsening_share", .5)):
            r = row(); r["short"][field] = value
            self.assertFalse(evidence(r)["passes"])
        r = row(); r["short_allocated_delta_W"] = -10
        self.assertFalse(evidence(r)["passes"])
        r = row(); r["longer"]["mean_delta_cf_mae"] = .02
        self.assertFalse(evidence(r)["passes"])

    def test_empty_group_is_insufficient_support(self):
        r = row()
        r["short"] = dict(sites=0, mean_delta_cf_mae=None, median_delta_cf_mae=None, worsening_share=None)
        self.assertFalse(evidence(r)["passes"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
