"""Hand-calculated identities for cohort errors and cancellation changes."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from b2_yearend_decomposition import decomposition,delta


class YearendContracts(unittest.TestCase):
    def test_improved_cohorts_can_worsen_total_by_losing_cancellation(self):
        old=decomposition([[10.]], [[-9.]], [1.])
        new=decomposition([[8.]], [[-5.]], [1.])
        self.assertEqual(old["W_all"],1.)
        self.assertEqual(new["W_all"],3.)
        d=delta(old,new)
        self.assertEqual(d["delta_W_fixed"],-2.)
        self.assertEqual(d["delta_W_rest"],-4.)
        self.assertEqual(d["delta_cross_cancellation"],-8.)
        self.assertEqual(d["delta_W_all"],2.)

    def test_within_cohort_cancellation_is_separate(self):
        r=decomposition([[5.,-3.]], [[-1.]], [2.])
        self.assertEqual(r["abs_individual_fixed"],16.)
        self.assertEqual(r["within_fixed"],12.)
        self.assertEqual(r["cross_cancellation"],4.)
        self.assertEqual(r["W_all"],2.)

    def test_opposite_groups_must_not_cancel_each_other(self):
        a=decomposition([[4.]],[[1.]],[1.])
        b=decomposition([[-4.]],[[-1.]],[1.])
        self.assertEqual(a["W_all"]+b["W_all"],10.)

    def test_zero_price_does_not_define_weighted_bias(self):
        r=decomposition([[4.]],[[1.]],[0.])
        self.assertEqual(r["W_all"],0.)
        self.assertIsNone(r["weighted_bias_all"])

    def test_invalid_values_and_mismatched_samples_rejected(self):
        with self.assertRaises(ValueError):
            decomposition([[np.nan]],[[1.]],[1.])
        with self.assertRaises(ValueError):
            decomposition([[1.]],[[1.]],[-1.])
        with self.assertRaises(ValueError):
            delta(decomposition([[1.]],[[1.]],[1.]),decomposition([[1.]],[[1.]],[2.]))


if __name__=="__main__":
    unittest.main()
