"""Small independent contracts for S0 leakage, joins and settlement arithmetic."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validation_core import (KST, WINDOWS, clock, grid, group_totals, join_forecasts,
                             local_score, prepare_forecast, truth_matrix, weights)


class TimeContracts(unittest.TestCase):
    def test_midnight_belongs_to_previous_day(self):
        c = clock(pd.DatetimeIndex(["2025-01-02 00:00"], tz=KST))
        self.assertEqual(str(c.target_day.iloc[0]), "2025-01-01 00:00:00+09:00")
        self.assertEqual(str(c.prediction_at.iloc[0]), "2024-12-31 17:00:00+09:00")

    def test_k_end_hour_boundaries_and_timezone(self):
        ts = pd.DatetimeIndex([f"2024-01-01 {h}:00" for h in (10, 11, 15, 16)], tz=KST)
        self.assertEqual(weights(ts).tolist(), [1, 3, 3, 1])
        self.assertEqual(weights(ts.tz_convert("UTC")).tolist(), [1, 3, 3, 1])

    def test_naive_or_nonhourly_target_rejected(self):
        for ts in (pd.DatetimeIndex(["2024-01-01"]),
                   pd.DatetimeIndex(["2024-01-01 01:00:00.001"], tz=KST)):
            with self.assertRaises(ValueError):
                clock(ts)

    def test_training_cutoff_inclusive_and_gap(self):
        for window in WINDOWS:
            cutoff = pd.Timestamp(window.cutoff, tz=KST)
            lower = pd.Timestamp(window.train_start, tz=KST)
            gen = pd.DataFrame({"ts": [lower, lower + pd.Timedelta(hours=1), cutoff,
                                       cutoff + pd.Timedelta(hours=1)]})
            self.assertEqual(window.training(gen).ts.tolist(), [gen.ts.iloc[1], cutoff])
            self.assertLess(cutoff, clock(window.axis()).prediction_at.min())
            self.assertEqual(clock(window.axis()).prediction_at.min() - cutoff, pd.Timedelta(hours=17))
        self.assertEqual([len(w.axis()) for w in WINDOWS], [8784, 3624])


class JoinContracts(unittest.TestCase):
    def setUp(self):
        self.axis = pd.date_range("2024-01-01 01:00", periods=2, freq="h", tz=KST)
        self.sites = pd.DataFrame({"site_id": ["x", "y"], "wa_station": ["a", "a"],
                                   "wb_station": ["b", "b"]})
        self.fa = pd.DataFrame({"wa_station": ["a"], "ts": [self.axis[0]], "round": [10], "fa": [4.]})
        self.fb = pd.DataFrame({"wb_station": ["b"], "ts": [self.axis[0]], "round": [10], "fb": [np.nan]})

    def test_missing_forecasts_preserve_grid_and_distinguish_nan(self):
        d = join_forecasts(grid(self.sites, self.axis), self.fa, self.fb, 10)
        self.assertEqual(len(d), 4)
        self.assertEqual(d.wa_row_present.sum(), 2)
        self.assertEqual(d.wb_row_present.sum(), 2)
        self.assertEqual(d.fb.isna().sum(), 4)
        d = join_forecasts(grid(self.sites, self.axis), self.fa, self.fb, 17)
        self.assertEqual(len(d), 4)
        self.assertFalse(d.wa_row_present.any())

    def test_duplicate_forecast_raises(self):
        with self.assertRaises(ValueError):
            join_forecasts(grid(self.sites, self.axis), pd.concat([self.fa, self.fa]), self.fb, 10)

    def test_nonhourly_forecast_never_rounded(self):
        fa = self.fa.copy()
        fa["ts"] += pd.Timedelta(milliseconds=1)
        clean, excluded = prepare_forecast(fa, "wa_station", ["fa"])
        self.assertEqual((len(clean), excluded), (0, 1))
        with self.assertRaises(ValueError):
            join_forecasts(grid(self.sites, self.axis), fa, self.fb, 10)

    def test_duplicate_generation_raises(self):
        gen = pd.DataFrame({"site_id": ["x", "x"], "ts": [self.axis[0]] * 2, "energy": [1., 2.]})
        with self.assertRaises(ValueError):
            truth_matrix(gen, ["x"], self.axis)


class ScoreContracts(unittest.TestCase):
    def setUp(self):
        self.axis = pd.date_range("2024-01-01 11:00", periods=2, freq="h", tz=KST)
        self.truth = pd.DataFrame([[10., 20., 30., 40.], [10., 20., 30., 40.]],
                                  index=self.axis, columns=list("abcd"))
        self.groups = pd.Series(["G1", "G1", "G2", "G2"], index=list("abcd"))
        self.price = pd.DataFrame({"ts": self.axis, "smp": [2., 0.]})

    def score(self, pred, truth=None, axis=None, price=None):
        return local_score(self.truth if truth is None else truth, pred, self.groups,
                           self.price if price is None else price, self.axis if axis is None else axis)

    def test_within_group_cancellation(self):
        pred = self.truth + [5., -5., 0., 0.]
        self.assertEqual(self.score(pred)["W_local"], 0.)

    def test_cross_group_errors_do_not_cancel(self):
        pred = self.truth + [5., 0., -5., 0.]
        score = self.score(pred)
        self.assertEqual(score["W_local"], 60.)  # 2 * 3 * (abs(5) + abs(-5))
        self.assertEqual(score["R_local"], 200.)
        self.assertEqual(score["S_local"], 194.)
        self.assertEqual(score["weighted_group_mae_kwh"], 5.)

    def test_missing_truth_excluded_before_sum(self):
        truth = self.truth.copy()
        truth.iloc[1, 0] = np.nan
        actual, complete = group_totals(truth, self.groups)
        self.assertEqual(complete.tolist(), [True, False])
        self.assertEqual(len(actual), 1)
        self.assertEqual(self.score(self.truth, truth, self.axis[:1])["times"], 1)
        with self.assertRaises(ValueError):
            self.score(self.truth, truth)

    def test_missing_or_negative_prediction_raises_even_at_zero_price(self):
        for value in (np.nan, np.inf, -1.):
            pred = self.truth.copy()
            pred.iloc[1, 0] = value
            with self.assertRaises(ValueError):
                self.score(pred)

    def test_raw_negative_truth_preserved(self):
        truth = self.truth.copy()
        truth.iloc[0, 0] = -10.
        score = self.score(self.truth, truth)
        self.assertEqual(score["W_local"], 120.)
        self.assertEqual(score["R_local"], 160.)

    def test_zero_price_denominator_undefined_and_empty_sample_rejected(self):
        price = self.price.assign(smp=0.)
        score = self.score(self.truth, price=price)
        self.assertEqual(score["W_local"], 0.)
        self.assertIsNone(score["weighted_group_mae_kwh"])
        with self.assertRaises(ValueError):
            self.score(self.truth, axis=self.axis[:0])

    def test_missing_duplicate_price_and_axis_mismatch_rejected(self):
        for price in (self.price.iloc[:1], pd.concat([self.price, self.price])):
            with self.assertRaises(ValueError):
                self.score(self.truth, price=price)
        with self.assertRaises(ValueError):
            self.score(self.truth.iloc[:1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
