"""Non-cancellation, signed-cost changes and daily concentration contracts."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from b3_summer_diagnostics import concentration, stats


class SummerDiagnostics(unittest.TestCase):
    def test_direction_change_and_group_non_cancellation(self):
        r=stats([5,-5],[7,-3],[2,2])
        self.assertEqual(r["B2_W"],20)
        self.assertEqual(r["B3_W"],20)
        self.assertEqual(r["delta_over_W"],4)
        self.assertEqual(r["delta_under_W"],-4)
        self.assertEqual(r["weighted_prediction_shift"],2)

    def test_zero_weight_is_retained_but_not_active(self):
        r=stats([10,0],[-20,0],[0,1])
        self.assertEqual(r["times"],2)
        self.assertEqual(r["B3_W"],0)
        self.assertEqual(r["positive_weight_times"],1)
        self.assertEqual(r["opposite_error_times"],0)

    def test_daily_gross_and_net_have_different_denominators(self):
        r=concentration([10,5,-12,0])
        self.assertEqual(r["gross_increase"],15)
        self.assertEqual(r["gross_improvement"],12)
        self.assertEqual(r["net_delta"],3)
        self.assertAlmostEqual(r["top_1_positive_days_share"],2/3)
        self.assertEqual(r["worse_days"],2)

    def test_invalid_and_zero_denominator(self):
        with self.assertRaises(ValueError): stats([1],[1],[-1])
        with self.assertRaises(ValueError): stats([float("nan")],[1],[1])
        with self.assertRaises(ValueError): concentration([])
        self.assertIsNone(stats([0],[0],[0])["change_pct"])


if __name__=="__main__": unittest.main()
