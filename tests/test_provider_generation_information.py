"""Check original-response sums of squares and correct partial regression."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from provider_generation_information import (extra_stats, extra_ss, evaluate, fit, mapping,
    rain_support, model_columns)
from conditional_support_date import prepare, cross, block_weights


def explicit_matrix(x, y, months, weights, date):
    d, s, h, p = x.shape
    z = np.concatenate([x, y[..., None]], axis=-1).reshape(-1, p + 1)
    group = np.repeat((months - 1) * s * h, s * h) + np.tile(np.arange(s * h), d)
    nuisance = np.eye(int(group.max()) + 1)[group]
    if date:
        ids = np.repeat(np.arange(d) * h, s * h) + np.tile(np.arange(h), d * s)
        nuisance = np.column_stack([nuisance, np.eye(d * h)[ids]])
    w = np.repeat(weights, s * h)
    active = w > 0
    a = nuisance[active] * np.sqrt(w[active, None])
    b = z[active] * np.sqrt(w[active, None])
    r = b - a @ np.linalg.lstsq(a, b, rcond=None)[0]
    return r.T @ r


class InformationChecks(unittest.TestCase):
    def test_extra_response_ss_restores_full_site_matrix_for_both_effects(self):
        rng = np.random.default_rng(12)
        pair = np.array([0, 1, 1, 2, 2, 2])
        sizes = np.bincount(pair)
        x = rng.normal(size=(18, 3, 2, 4))
        ysite = rng.normal(size=(18, 6, 2)) + rng.normal(size=(1, 6, 2))
        avg = np.stack([ysite[:, pair == i].mean(axis=1) for i in range(3)], axis=1)
        deviation = ysite - avg[:, pair]
        months = np.repeat([1, 2, 3], 6)
        weights = rng.integers(0, 5, size=18)
        weights[months == 2] = 0
        stats = prepare(x, avg, np.ones(avg.shape, dtype=bool), sizes, months)
        for date in [False, True]:
            got = cross(stats, weights, date)
            got[-1, -1] += extra_ss(extra_stats(deviation, months), weights)
            want = explicit_matrix(x[:, pair], ysite, months, weights, date)
            np.testing.assert_allclose(got, want, atol=1e-9)

    def test_partial_regression_residualizes_both_x_and_y(self):
        rng = np.random.default_rng(3)
        other = rng.normal(size=(1000, 2))
        target = other @ [2., -1.] + rng.normal(size=1000)
        y = .4 * target + other @ [1., 3.] + rng.normal(size=1000) * .2
        x = np.column_stack([other, target])
        z = np.column_stack([x, y])
        mat = z.T @ z
        result = fit(mat, [0, 1, 2])
        tx = target - other @ np.linalg.lstsq(other, target, rcond=None)[0]
        ty = y - other @ np.linalg.lstsq(other, y, rcond=None)[0]
        expected = 100 * (tx @ ty) / (tx @ tx)
        self.assertAlmostEqual(result['beta_pp'][2], expected, places=10)
        reduced = fit(mat, [0, 1])
        partial = (reduced['sse'] - result['sse']) / reduced['sse']
        self.assertAlmostEqual(partial, (tx @ ty) ** 2 / ((tx @ tx) * (ty @ ty)), places=10)

    def test_provider_nested_models_and_variance_denominators(self):
        rng = np.random.default_rng(6)
        x = rng.normal(size=(1000, 13))
        y = x[:, 2] * .2 + x[:, 8] * -.3 + x[:, 12] * .4 + rng.normal(size=1000)
        z = np.column_stack([x, y])
        matrix = z.T @ z
        rain = {'A': {'eligible': False}, 'B': {'eligible': False}}
        fits, metrics = evaluate(matrix, model_columns(rain), len(y))
        self.assertGreater(metrics['R2_AB'], metrics['R2_A'])
        self.assertGreater(metrics['R2_AB'], metrics['R2_B'])
        self.assertAlmostEqual(metrics['delta_R2_A_given_B'], metrics['R2_AB'] - metrics['R2_B'])
        self.assertAlmostEqual(metrics['A_cloud_partial_R2'], metrics['A_cloud_partial_correlation'] ** 2)
        self.assertNotIn('A_rain_delta_R2', metrics)
        others = x[:, [j for j in range(13) if j != 2]]
        residual = x[:, 2] - others @ np.linalg.lstsq(others, x[:, 2], rcond=None)[0]
        self.assertAlmostEqual(metrics['A_cloud_remaining_weather_sd'],
                               10 * np.sqrt(np.mean(residual ** 2)), places=10)

    def test_shared_source_matching_is_unique_and_maximal_in_known_graph(self):
        sites = pd.DataFrame({'wa_station': ['a', 'a', 'b', 'c', 'c'],
                              'wb_station': ['x', 'x', 'x', 'y', 'z']})
        pairs, inverse, counts, keep = mapping(sites)
        self.assertEqual(len(pairs), 4)
        self.assertEqual(keep.sum(), 2)
        self.assertTrue(pairs.loc[keep].wa_station.is_unique)
        self.assertTrue(pairs.loc[keep].wb_station.is_unique)
        self.assertEqual(counts.sum(), len(sites))
        np.testing.assert_array_equal(pairs.iloc[inverse].to_numpy(), sites.to_numpy())

    def test_no_rain_disagreement_is_not_sufficient_support(self):
        x = np.zeros((40, 6, 8, 13))
        pairs = pd.DataFrame({'wa_station': [f'a{i}' for i in range(6)], 'wb_station': [f'b{i}' for i in range(6)]})
        months = np.ones(40, dtype=int)
        result = rain_support(x, pairs, months)
        self.assertFalse(result['A']['eligible'])
        self.assertFalse(result['B']['eligible'])
        self.assertEqual(result['states']['A0_B0']['pair_hour_rows'], 40 * 6 * 8)

    def test_rain_support_counts_unique_pairs_and_cells(self):
        x = np.zeros((40, 6, 8, 13))
        x[:20, ..., 4] = 1
        pairs = pd.DataFrame({'wa_station': [f'a{i}' for i in range(6)], 'wb_station': [f'b{i}' for i in range(6)]})
        result = rain_support(x, pairs, np.ones(40, dtype=int))
        self.assertTrue(result['A']['eligible'])
        self.assertEqual(result['A']['supported_cells'], 48)
        self.assertFalse(result['B']['eligible'])

    def test_rank_deficiency_is_not_silently_regularized(self):
        x = np.ones((3, 3))
        self.assertFalse(fit(x, [0, 1])['valid'])

    def test_extra_ss_is_zero_for_single_site_per_pair(self):
        stats = extra_stats(np.zeros((30, 5, 8)), np.repeat([1, 2, 3], 10))
        self.assertEqual(extra_ss(stats, block_weights(30, 7, np.random.default_rng(8))), 0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
