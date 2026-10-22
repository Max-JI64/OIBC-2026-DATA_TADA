"""Paired target-day blocks and removal sensitivity without data or models."""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from p010_block_stability import block_summary


class Blocks(unittest.TestCase):
    def test_interval_end_and_partial_final_block(self):
        axis = pd.date_range("2024-07-01 01:00", periods=16*24, freq="h", tz="Asia/Seoul")
        b = np.ones(len(axis)); p = b*.9
        r = block_summary(axis, "2024-07-01", b, p, 7)
        self.assertEqual([z["days"] for z in r["rows"]], [7, 7, 2])
        self.assertEqual(r["rows"][0]["last_target"], "2024-07-07 23:00:00+09:00")
        self.assertEqual(r["removal_reversal_count"], 0)

    def test_positive_remainder_and_zero_denominator(self):
        axis = pd.date_range("2024-07-01 01:00", periods=14*24, freq="h", tz="Asia/Seoul")
        b = np.ones(len(axis))*10
        p = b.copy(); p[:168] -= 2; p[168:] += 1
        r = block_summary(axis, "2024-07-01", b, p, 7)
        self.assertEqual(r["removal_reversal_count"], 1)
        zero = block_summary(axis[:24], "2024-07-01", b[:24]*0, p[:24]*0, 7)
        self.assertIsNone(zero["rows"][0]["remaining_change_pct"])
        self.assertEqual(zero["rows"][0]["remaining_direction"], 0)

    def test_invalid_costs_and_unregistered_axis_rejected(self):
        axis = pd.date_range("2024-07-01 01:00", periods=24, freq="h", tz="Asia/Seoul")
        with self.assertRaises(ValueError):
            block_summary(axis, "2024-07-01", np.ones(24), np.ones(24)*np.nan, 7)
        with self.assertRaises(ValueError):
            block_summary(axis, "2024-07-01", np.ones(24), np.ones(24), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
