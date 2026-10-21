"""Analytical checks for dependence-preserving nulls and FA calculations."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from weather_latent_structure import (align_loadings, balanced_cube, block_indices,
    circular_cross_products, correlation_from_stats, daily_stats, diagnostics,
    fit_factor, make_view, max_factors, null_eigenvalues)


class LatentStructureChecks(unittest.TestCase):
    def test_fft_matches_direct_shifts(self):
        x = np.random.default_rng(11).normal(size=(19, 3, 2, 4))
        cross = circular_cross_products(x)
        shifts = np.array([2, 8, 1, 5])
        shifted = np.stack([np.roll(x[..., j], shifts[j], axis=0) for j in range(4)], axis=-1)
        direct = shifted.reshape(-1, 4).T @ shifted.reshape(-1, 4)
        got = cross[(shifts[:, None] - shifts[None, :]) % 19, np.arange(4)[:, None], np.arange(4)[None, :]]
        np.testing.assert_allclose(got, direct, atol=1e-10)

    def test_daily_statistics_match_direct_covariance(self):
        x = np.random.default_rng(3).normal(size=(20, 3, 4, 5)) + np.arange(5)
        sums, crosses, cells = daily_stats(x)
        r = correlation_from_stats(sums.sum(0), crosses.sum(0), len(x) * cells)
        np.testing.assert_allclose(r, np.corrcoef(x.reshape(-1, 5), rowvar=False), atol=1e-12)

    def test_known_two_factor_covariance(self):
        load = np.zeros((8, 2))
        load[:4, 0] = [.8, .75, .7, .65]
        load[4:, 1] = [.8, .75, .7, .65]
        r = load @ load.T + np.diag(1 - (load ** 2).sum(1))
        fit = fit_factor(r, 2)
        self.assertTrue(fit['valid'], fit)
        _, congruence = align_loadings(load, fit['loadings'])
        self.assertGreater(min(congruence), .999)
        self.assertLess(fit['rms_residual'], 1e-6)

    def test_sign_and_permutation_alignment(self):
        a = np.random.default_rng(4).normal(size=(8, 3))
        b = a[:, [2, 0, 1]] * [-1, 1, -1]
        aligned, c = align_loadings(a, b)
        np.testing.assert_allclose(aligned, a)
        np.testing.assert_allclose(c, 1)

    def test_invalid_correlation_is_rejected(self):
        r = np.ones((6, 6))
        self.assertFalse(diagnostics(r)['eligible'])
        self.assertFalse(fit_factor(r, 2)['valid'])
        self.assertEqual(max_factors(6), 2)
        self.assertEqual(max_factors(7), 3)

    def test_centering_and_null_diagonal(self):
        rng = np.random.default_rng(7)
        months = np.repeat(np.arange(1, 13), 16)
        x = rng.normal(size=(192, 2, 24, 4))
        x += months[:, None, None, None] * np.arange(1, 5)
        z = make_view(x, months, 'within_day')
        for m in range(1, 13):
            np.testing.assert_allclose(z[months == m].mean(0), 0, atol=1e-12)
        ev = null_eigenvalues(z, months, rng, reps=9)
        np.testing.assert_allclose(ev.sum(1), 4, atol=1e-10)

    def test_bootstrap_keeps_seven_day_runs(self):
        idx = block_indices(365, np.random.default_rng(3))
        self.assertEqual(len(idx), 365)
        for start in range(0, 350, 7):
            np.testing.assert_array_equal(np.diff(idx[start:start+7]) % 365, 1)

    def test_conflicting_duplicate_removes_entire_balanced_date(self):
        ts = pd.date_range('2024-01-01', '2025-01-01', freq='h', tz='Asia/Seoul')[1:]
        rows = pd.DataFrame({'ts': ts, 'wa_station': 'WA_TEST', 'wa_temp': 15., 'wa_humidity': 50.,
                             'wa_cloud': 30., 'wa_ground_press': 1000., 'wa_wind_speed': 2.,
                             'wa_rain': 0., 'wa_snow': 0.})
        duplicate = rows.iloc[[10]].copy()
        duplicate['wa_temp'] = 20.
        rows = pd.concat([rows, duplicate], ignore_index=True)
        cube, _, _, audit = balanced_cube(rows, 'wa', 'observed', ['WA_TEST'], 2024)
        self.assertEqual(audit['conflicting_cells'], 1)
        self.assertEqual(audit['balanced_dates'], 365)
        self.assertEqual(cube.shape, (365, 1, 24, 7))


if __name__ == '__main__':
    unittest.main()
