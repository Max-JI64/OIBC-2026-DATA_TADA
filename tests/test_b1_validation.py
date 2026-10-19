"""Registered one-column ablation and V1 decision boundaries."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from b0_v1_baseline import experiment_features
from b1_v1_compare import compare_results
from s0_validation_check import baseline_contract


def scores(annual, quarters):
    return [dict(sample="fixed_148", period=period, times=1, first_ts="a", last_ts="b", smp_sum=1.,
                 zero_price_times=0, price_k_sum=1., R_local=100., W_local=w,
                 weighted_group_mae_kwh=w/2, S_local=100.-.1*w)
            for period, w in zip(("all", "Q1", "Q2", "Q3", "Q4"), (annual, *quarters))]


class PressureSelection(unittest.TestCase):
    def test_only_pressure_is_removed_and_order_preserved(self):
        spec = baseline_contract()
        self.assertEqual(experiment_features(spec, "B0_V1"), spec["FEATS"])
        b1 = experiment_features(spec, "B1_V1")
        self.assertEqual(len(b1), 23)
        self.assertEqual(b1, [c for c in spec["FEATS"] if c != "wa_ground_press"])

    def test_improvement_and_exact_five_percent_guard(self):
        result = compare_results(scores(400, [100]*4), scores(390, [105, 95, 95, 95]))
        self.assertEqual(result["selected_for_V2"], "B1")

    def test_more_than_five_percent_retains_b0(self):
        result = compare_results(scores(400, [100]*4), scores(390, [105.01, 94.99, 95, 95]))
        self.assertEqual(result["selected_for_V2"], "B0")
        self.assertEqual(result["unstable_quarters"], ["Q1"])

    def test_no_improvement_or_tie_retains_b0(self):
        for annual in (400, 401):
            self.assertEqual(compare_results(scores(400, [100]*4), scores(annual, [100]*4))["selected_for_V2"], "B0")

    def test_zero_quarter_reports_absolute_difference(self):
        result = compare_results(scores(300, [0, 100, 100, 100]), scores(290, [1, 89, 100, 100]))
        self.assertEqual(result["selected_for_V2"], "B0")
        self.assertIsNone(result["rows"][1]["change_pct"])

    def test_different_truth_or_weight_is_rejected(self):
        old, new = scores(400, [100]*4), scores(390, [95, 95, 100, 100])
        new[0]["price_k_sum"] = 2.
        with self.assertRaises(ValueError):
            compare_results(old, new)


if __name__ == "__main__":
    unittest.main()
