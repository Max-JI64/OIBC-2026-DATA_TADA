"""V2 boundary, feature freeze and empty diagnostic cohort contracts."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from b0_v1_baseline import experiment_features, individual_metrics
from s0_validation_check import baseline_contract
from validation_core import KST, WINDOWS, clock


class V2Contracts(unittest.TestCase):
    def test_selected_b0_keeps_all_24_features(self):
        spec = baseline_contract()
        self.assertEqual(experiment_features(spec, "B0_V2"), experiment_features(spec, "B0_V1"))
        with self.assertRaises(ValueError):
            experiment_features(spec, "B1_V2")

    def test_v2_training_cutoff_and_target_midnight(self):
        w = WINDOWS[1]
        self.assertEqual(w.name, "V2")
        self.assertEqual(len(w.axis()), 3624)
        self.assertEqual(str(w.axis()[-1]), "2025-06-01 00:00:00+09:00")
        gen = pd.DataFrame({"ts": pd.DatetimeIndex(["2023-01-01 00:00", "2023-01-01 01:00",
                     "2024-12-31 00:00", "2024-12-31 01:00"], tz=KST)})
        self.assertEqual(w.training(gen).ts.tolist(), gen.ts.iloc[[1, 2]].tolist())
        self.assertEqual(clock(w.axis()).prediction_at.min() - pd.Timestamp(w.cutoff, tz=KST), pd.Timedelta(hours=17))

    def test_empty_months_and_no_train_cohort_are_not_fabricated(self):
        axis = pd.date_range("2025-03-01 11:00", periods=2, freq="h", tz=KST)
        sites = pd.DataFrame({"site_id": ["x", "y"], "capacity_kw": [100., 200.], "lat": [35., 36.], "lon": [127., 128.]})
        truth = pd.DataFrame([[50., 100.], [60., 120.]], index=axis, columns=sites.site_id)
        spec = {"solar": lambda ts, lat, lon: (np.ones(len(ts)), np.zeros(len(ts)), np.ones(len(ts)))}
        with tempfile.TemporaryDirectory() as folder:
            rows = individual_metrics(truth, truth.copy(), sites, ["x", "y"], spec, output=Path(folder))
        labels = [r["scope"] for r in rows]
        self.assertIn("has_train", labels)
        self.assertNotIn("no_train", labels)
        self.assertNotIn("solar_night", labels)
        self.assertNotIn("M06", labels)
        self.assertEqual(next(r for r in rows if r["scope"] == "all")["kwh_mae"], 0.)


if __name__ == "__main__":
    unittest.main()
