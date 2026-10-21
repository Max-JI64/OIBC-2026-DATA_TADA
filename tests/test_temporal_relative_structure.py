"""Synthetic safeguards for peer comparisons and scan calibration."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from temporal_relative_structure import (seasonal_design, relative_to_peers, scan,
                                        circular_indices, global_pvalues, persistence)


class TemporalTests(unittest.TestCase):
    def test_leave_one_out_and_singleton(self):
        y = np.array([[1., 2., 6., 9.], [3., 4., 8., 10.]])
        r, ok = relative_to_peers(y, np.array(['a', 'a', 'a', 'b']))
        np.testing.assert_array_equal(ok, [True, True, True, False])
        np.testing.assert_allclose(r[0], [-3., -1.5, 4.5])
        np.testing.assert_allclose(r.sum(axis=1), 0)

    def test_common_weather_cancels(self):
        rng = np.random.default_rng(1)
        y = rng.normal(size=(100, 4))
        stations = np.array(['a', 'a', 'b', 'b'])
        added = np.column_stack([np.sin(np.arange(100))] * 2 + [np.cos(np.arange(100))] * 2)
        np.testing.assert_allclose(relative_to_peers(y, stations)[0],
                                   relative_to_peers(y + added, stations)[0], atol=1e-14)

    def test_repeated_season_removed(self):
        dates = pd.date_range('2023-01-01', '2024-12-31')
        x = seasonal_design(dates, 3)
        y = x @ np.arange(7.)
        np.testing.assert_allclose(y - x @ np.linalg.lstsq(x, y, rcond=None)[0], 0, atol=1e-13)

    def test_leap_phase(self):
        x = seasonal_design(pd.to_datetime(['2023-01-01', '2024-01-01']), 6)
        np.testing.assert_array_equal(x[0], x[1])

    def test_step_date_and_sign(self):
        y = np.r_[np.zeros(200), np.ones(300)][:, None]
        cuts, values, k, peak, delta = scan(y)
        self.assertEqual(k[0], 200)
        self.assertGreater(peak[0], 0)
        self.assertEqual(delta[0], 1.)
        np.testing.assert_allclose(scan(-y)[1], values)
        self.assertEqual(cuts[0], 90)
        self.assertEqual(cuts[-1], 410)

    def test_constant_and_short(self):
        np.testing.assert_array_equal(scan(np.ones((200, 2)))[3], [0., 0.])
        with self.assertRaises(ValueError):
            scan(np.ones((179, 2)))

    def test_joint_indices(self):
        ix = circular_indices(731, 14, np.random.default_rng(2))
        self.assertEqual(len(ix), 731)
        self.assertTrue(((ix >= 0) & (ix < 731)).all())
        for start in range(0, 731, 14):
            np.testing.assert_array_equal(np.diff(ix[start:start+14]) % 731, 1)

    def test_global_adjustment(self):
        np.testing.assert_allclose(global_pvalues(np.array([1., 3., 4.]), np.array([1., 2., 3.])), [1., .5, .25])

    def test_persistence_rejects_small_or_reversing(self):
        y = np.column_stack([np.r_[np.zeros(200), np.full(300, .03)],
                             np.r_[np.zeros(200), np.full(300, .01)]])
        _, _, passes = persistence(y, [200, 200], np.array([.03, .01]))
        np.testing.assert_array_equal(passes, [True, False])
        _, _, passes = persistence(y, [200, 200], np.array([-.03, -.01]))
        self.assertFalse(passes.any())


if __name__ == '__main__':
    unittest.main()
