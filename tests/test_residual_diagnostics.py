"""Independent examples for cancellation, direction and fixed round averaging."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from v1_residual_diagnostics import residual_summary, round_summary


class ResidualContracts(unittest.TestCase):
    def test_within_group_cancellation_and_direction(self):
        r = residual_summary([0, 4, -2], [10, 4, 6], [1, 2, 3])
        self.assertEqual(r["W"], 14)
        self.assertEqual(r["over_W"], 8)
        self.assertEqual(r["under_W"], 6)
        self.assertAlmostEqual(r["cancellation_ratio"], 1-14/36)

    def test_opposite_round_errors_cancel(self):
        r = round_summary([5., -5.], [-5., 5.], [2., 2.])
        self.assertEqual(r["Wmean"], 0)
        self.assertEqual(r["averaging_gain_pct"], 100)
        self.assertEqual(r["opposite_sign_fraction"], 1)

    def test_same_sign_round_errors_do_not_gain(self):
        r = round_summary([2., 4.], [6., 8.], [1., 3.])
        self.assertEqual(r["averaging_gain_pct"], 0)
        self.assertEqual(r["opposite_sign_fraction"], 0)

    def test_zero_weights_have_undefined_weighted_metrics(self):
        r = residual_summary([2., -1.], [2., 1.], [0., 0.])
        self.assertIsNone(r["weighted_mae_kwh"])
        self.assertIsNone(r["cancellation_ratio"])
        self.assertIsNone(r["top_1pct_W_share"])
        self.assertEqual(r["mae_kwh"], 1.5)

    def test_nonfinite_or_negative_weights_rejected(self):
        for q in ([1., -1.], [1., np.nan]):
            with self.assertRaises(ValueError):
                residual_summary([1., 2.], [1., 2.], q)

    def test_individual_sum_cannot_be_below_group_error(self):
        with self.assertRaises(ValueError):
            residual_summary([10.], [2.], [1.])


if __name__ == "__main__":
    unittest.main()
