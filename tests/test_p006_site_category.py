"""P-006 feature, leakage, gate and sparse diagnostic contracts; no fitting."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from p006_site_category import categorical_features, categories_from_training, gate, diagnostics, NEW
from validation_core import Window, KST


def original(n=4):
    frame = pd.DataFrame(np.arange(n*39, dtype="float32").reshape(n, 39), columns=[f"x{i}" for i in range(39)])
    frame.iloc[0, 0] = np.nan
    return frame


def score_rows(total=100., part=100., annual=True):
    periods = [f"Q{i}" for i in range(1, 5)] if annual else ["M07", "M08", "M09"]
    return [dict(sample="fixed_cohort", period=p, times=1, first_ts="a", last_ts="b", R_local=1000.,
                 price_k_sum=1., W_local=w) for p, w in [("all", total)] + [(p, part) for p in periods]]


class CategoryContracts(unittest.TestCase):
    def test_values_schema_and_unknown_mask(self):
        x = original()
        ids = pd.Series(["b", "a", "new", "b"])
        y = categorical_features(x, ids, ["a", "b"])
        pd.testing.assert_frame_equal(x, y.iloc[:, :39])
        self.assertEqual(y.shape, (4, 40))
        self.assertEqual(y[NEW].cat.codes.tolist(), [1, 0, -1, 1])
        self.assertFalse(y[NEW].cat.ordered)

    def test_row_mismatch_and_numeric_original_rejected(self):
        x, ids = original(), pd.Series(["a"] * 4)
        with self.assertRaises(ValueError):
            categorical_features(x, ids.iloc[::-1], ["a"])
        with self.assertRaises(ValueError):
            categorical_features(x.astype("float64"), ids, ["a"])
        with self.assertRaises(ValueError):
            categorical_features(x, ids, ["b", "a"])

    def test_dictionary_and_batch_order(self):
        ids = pd.Series(["b", "a", "c", "a"])
        categories = categories_from_training(ids)
        self.assertEqual(categories, ["a", "b", "c"])
        x = original()
        full = categorical_features(x, ids, categories)
        pieces = pd.concat([categorical_features(x.iloc[:2], ids.iloc[:2], categories),
                            categorical_features(x.iloc[2:], ids.iloc[2:], categories)])
        pd.testing.assert_frame_equal(full, pieces)
        reverse = categorical_features(x.iloc[::-1], ids.iloc[::-1], categories).iloc[::-1]
        pd.testing.assert_frame_equal(full, reverse)

    def test_future_and_night_only_not_in_training_dictionary(self):
        w = Window("test", "2024-01-01", "2024-01-02", "2023-01-01", "2023-12-31")
        g = pd.DataFrame({"ts": pd.DatetimeIndex(["2023-12-30 12:00", "2023-12-31 01:00", "2023-12-30 23:00"], tz=KST),
                          "site_id": ["seen", "future", "night"], "elev": [10., 10., -1.]})
        eligible = w.training(g)
        self.assertEqual(categories_from_training(eligible.loc[eligible.elev.gt(0), "site_id"]), ["seen"])

    def test_missing_ids_and_empty_dictionary_rejected(self):
        with self.assertRaises(ValueError):
            categories_from_training(pd.Series([None]))
        with self.assertRaises(ValueError):
            categorical_features(original(), pd.Series(["a"] * 4), [])


class DecisionContracts(unittest.TestCase):
    def test_strict_improvement_and_five_percent(self):
        self.assertTrue(gate(score_rows(), score_rows(90, 105), True)["passes"])
        self.assertFalse(gate(score_rows(), score_rows(90, 105.001), True)["passes"])
        self.assertFalse(gate(score_rows(), score_rows(100*(1-1e-13), 90), True)["passes"])
        self.assertFalse(gate(score_rows(0, 0), score_rows(0, 0), True)["passes"])

    def test_rolling_months_and_scope(self):
        a, b = score_rows(annual=False), score_rows(90, 105, annual=False)
        self.assertTrue(gate(a, b, False)["passes"])
        b[0]["times"] = 2
        with self.assertRaises(ValueError):
            gate(a, b, False)

    def test_one_sided_sparse_unknowns_are_not_zero_filled(self):
        axis = pd.date_range("2024-07-01 11:00", periods=2, freq="h", tz=KST)
        truth = pd.DataFrame([[10., 10., np.nan], [10., 10., 5.]], index=axis, columns=["a", "b", "c"])
        pred = pd.DataFrame([[12., 9., 50.], [12., 9., 7.]], index=axis, columns=truth.columns)
        sites = pd.DataFrame({"site_id": ["a", "b", "c"], "current_group": ["A", "B", "A"]})
        price = pd.DataFrame({"ts": axis, "smp": [1., 1.]})
        with patch("p006_site_category.cohort_ids", return_value=["a"]):
            result = diagnostics(truth, pred, sites, price, ["a", "b"], ["a"])
        self.assertFalse(next(r for r in result["cohort_coverage"] if r["cohort"] == "train_unseen")["W_available"])
        unknown = next(r for r in result["observed_id_errors"] if r["category"] == "unseen")
        self.assertEqual(unknown["observed_site_hours"], 3)
        self.assertAlmostEqual(unknown["individual_mae_kwh"], 4/3)
        whole = sum(r["W"] for r in result["group_rows"] if r["cohort"] == "complete" and r["k"] == "all")
        self.assertEqual(whole, 18.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
