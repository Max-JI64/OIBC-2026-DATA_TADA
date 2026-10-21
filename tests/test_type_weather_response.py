"""Checks fixed types, same-station contrasts and original-site coefficient equivalence."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from type_weather_response import group_mapping, full_design, shared_data, metrics, ref_scale
from conditional_support_date import prepare, cross, solve, design
from verify_conditional_support_date import direct_cross


class TypeResponseChecks(unittest.TestCase):
    def setUp(self):
        self.sites = pd.DataFrame({'site_id': [f's{i}' for i in range(8)],
            'wb_station': ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c'], 'capacity_kw': np.ones(8)})
        self.labels = np.array([0, 0, 1, 0, 1, 1, 0, 1])
        rng = np.random.default_rng(8)
        raw = rng.uniform(size=(30, 3, 2, 6))
        raw[..., 0:2] *= 100
        raw[..., 2] *= 30
        raw[..., 3:5] *= 5
        raw[..., 5] = np.where(raw[..., 5] > .6, raw[..., 5], 0.)
        self.weather = raw[:, [0, 0, 0, 1, 1, 1, 2, 2]]
        self.response = rng.normal(size=(30, 8, 2))
        self.months = np.repeat([1, 2, 3], 10)
        self.weights = rng.integers(0, 4, size=30)

    def test_full_design_keeps_only_assigned_type_block(self):
        x = full_design(self.weather, self.labels)
        np.testing.assert_allclose(x[:, self.labels == 0, :, :7], design(self.weather[:, self.labels == 0]))
        self.assertEqual(np.count_nonzero(x[:, self.labels == 0, :, 7:]), 0)
        self.assertEqual(np.count_nonzero(x[:, self.labels == 1, :, :7]), 0)

    def test_grouping_preserves_original_site_weather_coefficients(self):
        keys, inverse, groups, _, _ = group_mapping(self.sites, self.labels)
        x = full_design(self.weather, self.labels)
        gx = np.stack([x[:, g[0]] for g in groups], axis=1)
        gy = np.stack([self.response[:, g].mean(axis=1) for g in groups], axis=1)
        stats = prepare(gx, gy, np.ones(gy.shape, dtype=bool), np.array([len(g) for g in groups]), self.months)
        for date in [False, True]:
            got = cross(stats, self.weights, date)
            expected = direct_cross(x, self.response, np.ones(self.response.shape, dtype=bool),
                                    np.ones(8), self.months, self.weights, date)
            np.testing.assert_allclose(got[:-1], expected[:-1], atol=1e-8)
            a, b = solve(got), solve(expected)
            self.assertTrue(a['valid'] and b['valid'])
            np.testing.assert_allclose(a['beta_pp'], b['beta_pp'], atol=1e-7)

    def test_same_station_gap_uses_both_type_means_and_sign(self):
        x, y, _ = shared_data(self.weather, self.response, self.sites, self.labels, ['a', 'b', 'c'])
        np.testing.assert_allclose(y[:, 0], self.response[:, 2] - self.response[:, [0, 1]].mean(axis=1))
        _, reversed_y, _ = shared_data(self.weather, self.response, self.sites, 1 - self.labels, ['a', 'b', 'c'])
        np.testing.assert_allclose(reversed_y, -y)

    def test_single_type_station_cannot_enter_gap_comparison(self):
        labels = self.labels.copy()
        labels[-2:] = 0
        _, _, _, mixed, sturdy = group_mapping(self.sites, labels)
        self.assertNotIn('c', mixed)
        with self.assertRaises(ValueError):
            shared_data(self.weather, self.response, self.sites, labels, ['c'])

    def test_sturdy_station_requires_two_sites_in_each_type(self):
        sites = pd.DataFrame({'wb_station': ['a'] * 4 + ['b'] * 3})
        labels = np.array([0, 0, 1, 1, 0, 0, 1])
        _, _, _, mixed, sturdy = group_mapping(sites, labels)
        self.assertEqual(mixed, ['a', 'b'])
        self.assertEqual(sturdy, ['a'])

    def test_delta_metrics_do_not_confuse_type_slope_and_difference(self):
        result = {'valid': True, 'beta_pp': list(range(14))}
        got = metrics(result, 'all_absolute')
        self.assertEqual(got['C1_cloud'], 0)
        self.assertEqual(got['C2_cloud'], 7)
        self.assertEqual(got['delta_cloud'], 7)
        self.assertEqual(metrics({'valid': True, 'beta_pp': list(range(7))}, 'shared')['delta_humidity'], 1)
        self.assertEqual(metrics({'valid': False}, 'shared'), {})

    def test_frozen_reference_is_independent_of_weather_availability(self):
        sites = self.sites.iloc[:2].copy()
        days = pd.date_range('2023-01-01', '2023-12-31', tz='Asia/Seoul')
        times = pd.DatetimeIndex([d + pd.Timedelta(hours=h + 1) for d in days for _ in range(2) for h in range(9, 17)])
        gen = pd.DataFrame({'site_id': np.tile(np.repeat(sites.site_id, 8), len(days)), 'ts': times,
                            'energy': np.tile(np.repeat([.2, .4], 8), len(days))})
        np.testing.assert_allclose(ref_scale(gen, sites), [.2, .4], atol=1e-14)
        with self.assertRaises(ValueError):
            ref_scale(gen.iloc[1:], sites)

    def test_sitewise_rescaling_precedes_type_averaging(self):
        response = np.array([[[1.], [2.], [3.]]])
        sites = pd.DataFrame({'wb_station': ['a', 'a', 'a']})
        weather = np.ones((1, 3, 1, 6))
        labels = np.array([0, 0, 1])
        relative = response / np.array([1., 2., 1.])[None, :, None]
        _, gap, _ = shared_data(weather, relative, sites, labels, ['a'])
        self.assertAlmostEqual(float(gap[0, 0, 0]), 2.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
