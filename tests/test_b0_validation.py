"""B0 transformation contracts independent of LightGBM training."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from b0_v1_baseline import prediction_values, target_values


class BaselineTransforms(unittest.TestCase):
    def test_target_clip_does_not_change_truth(self):
        d = pd.DataFrame({"energy": [-5., 0., 50., 150.], "cap": [100.] * 4})
        before = d.copy(deep=True)
        np.testing.assert_array_equal(target_values(d), np.array([0, 0, .5, 1.2], dtype="float32"))
        pd.testing.assert_frame_equal(d, before)

    def test_night_zero_negative_clip_and_no_upper_prediction_clip(self):
        d = pd.DataFrame({"elev": [-1., 0., 10., 20.], "cap": [100.] * 4})
        np.testing.assert_array_equal(prediction_values(d, np.array([1., 2., -.2, 1.5])), [0., 0., 0., 150.])

    def test_nonfinite_prediction_rejected_even_at_night(self):
        d = pd.DataFrame({"elev": [-1.], "cap": [100.]})
        with self.assertRaises(ValueError):
            prediction_values(d, np.array([np.nan]))

    def test_average_is_taken_after_each_round_clipping(self):
        d = pd.DataFrame({"elev": [10.], "cap": [100.]})
        p = (prediction_values(d, np.array([-.4])) + prediction_values(d, np.array([.2]))) / 2
        np.testing.assert_array_equal(p, [10.])


if __name__ == "__main__":
    unittest.main()
