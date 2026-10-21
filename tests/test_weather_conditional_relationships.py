"""Check spline shape, exact weighted fixed effects and conditional contrasts."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from weather_conditional_relationships import (CONTRASTS, absorb, block_weights,
    contrast_vector, design, prepare_stats, rcs_basis, solve_cross, support_mask,
    support_stats, within_cross)


class ConditionalChecks(unittest.TestCase):
    def test_spline_has_linear_tails_and_curved_interior(self):
        knots = [0, 25, 50, 75, 100]
        left = rcs_basis(np.array([-30., -20., -10.]), knots)
        right = rcs_basis(np.array([110., 120., 130.]), knots)
        middle = rcs_basis(np.array([30., 40., 50.]), knots)
        np.testing.assert_allclose(np.diff(left, n=2, axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(np.diff(right, n=2, axis=0), 0, atol=1e-12)
        self.assertGreater(np.abs(np.diff(middle, n=2, axis=0)).max(), 1e-4)

    def test_weighted_absorption_matches_explicit_dummy_ols(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(24, 2, 2, 3))
        months = np.repeat([1, 2], 12)
        offsets = rng.normal(size=(2, 2, 2))
        y = np.einsum('dshp,p->dsh', x, [.2, -.3, .4]) + offsets[months - 1]
        y += rng.normal(size=y.shape) * .1
        weights = rng.integers(0, 4, size=24)
        stats = prepare_stats(x, y, months)
        xx, xy, yy = within_cross(stats, weights)
        got = solve_cross(xx, xy, yy)
        self.assertTrue(got['valid'])
        group = np.repeat((months - 1) * 4, 4) + np.tile(np.arange(4), 24)
        dummy = np.eye(8)[group]
        complete = np.column_stack([x.reshape(-1, 3), dummy])
        w = np.sqrt(np.repeat(weights, 4))
        want = np.linalg.lstsq(complete * w[:, None], y.ravel() * w, rcond=None)[0][:3]
        np.testing.assert_allclose(got['beta'], want, atol=1e-10)

    def test_unweighted_stats_match_residualized_design(self):
        rng = np.random.default_rng(6)
        x = rng.normal(size=(30, 2, 3, 4)) + np.arange(4)
        y = rng.normal(size=(30, 2, 3)) + np.arange(30)[:, None, None]
        months = np.repeat([1, 2, 3], 10)
        xx, xy, yy = within_cross(prepare_stats(x, y, months), np.ones(30))
        cx = absorb(x, months).reshape(-1, 4)
        cy = absorb(y, months).ravel()
        np.testing.assert_allclose(xx, cx.T @ cx, atol=1e-9)
        np.testing.assert_allclose(xy, cx.T @ cy, atol=1e-9)
        np.testing.assert_allclose(yy, cy @ cy, atol=1e-9)

    def test_bootstrap_recomputes_weighted_cell_means(self):
        x = np.arange(12., dtype=float).reshape(12, 1, 1, 1) ** 2
        y = np.arange(12.).reshape(12, 1, 1) ** 3
        months = np.repeat([1, 2], 6)
        w = np.array([3, 1, 0, 0, 0, 1, 3, 1, 0, 0, 0, 1])
        xx, xy, yy = within_cross(prepare_stats(x, y, months), w)
        cx = absorb(x, months, w).reshape(-1, 1)
        cy = absorb(y, months, w).ravel()
        np.testing.assert_allclose(xx, cx.T @ (cx * w[:, None]))
        np.testing.assert_allclose(xy, cx.T @ (cy * w))
        np.testing.assert_allclose(yy, (cy * w) @ cy)

    def test_empty_resampled_month_matches_weighted_dummy_ols(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(20, 2, 2, 3))
        months = np.repeat([1, 2], 10)
        y = rng.normal(size=(20, 2, 2))
        weights = np.r_[np.zeros(10), np.ones(10)]
        xx, xy, yy = within_cross(prepare_stats(x, y, months), weights)
        fit = solve_cross(xx, xy, yy)
        group = np.repeat((months - 1) * 4, 4) + np.tile(np.arange(4), 20)
        complete = np.column_stack([x.reshape(-1, 3), np.eye(8)[group]])
        w = np.sqrt(np.repeat(weights, 4))
        want = np.linalg.lstsq(complete * w[:, None], y.ravel() * w, rcond=None)[0][:3]
        self.assertTrue(fit['valid'])
        np.testing.assert_allclose(fit['beta'], want, atol=1e-10)

    def test_rain_cloud_interaction_contrast(self):
        weather = np.array([50., 60., 20., 3., 3., 0.])
        _, names, _ = design(weather)
        beta = np.zeros(len(names))
        beta[names.index('wet_cloud')] = 1
        c25 = next(c for c in CONTRASTS if c['id'] == 'rain_cloud25')
        c75 = next(c for c in CONTRASTS if c['id'] == 'rain_cloud75')
        self.assertAlmostEqual(contrast_vector(c25) @ beta, -50.)
        self.assertAlmostEqual(contrast_vector(c75) @ beta, 50.)

    def test_invalid_basis_and_constant_response_are_rejected(self):
        self.assertFalse(solve_cross(np.ones((3, 3)), np.ones(3), 10)['valid'])
        self.assertFalse(solve_cross(np.eye(3), np.zeros(3), 0)['valid'])

    def test_support_and_rain_threshold(self):
        w = np.zeros((30, 12, 8, 6))
        w[..., 0] = 25
        w[..., 5] = .05
        good = support_stats(support_mask(w, {'cloud': 25., 'rain': 0.}))
        self.assertTrue(good['supported'])
        bad = support_stats(support_mask(w, {'cloud': 75., 'rain': 1.}))
        self.assertFalse(bad['supported'])
        x, _, groups = design(w)
        np.testing.assert_array_equal(x[..., groups['rain']], 0)

    def test_block_resampling_preserves_date_count(self):
        w = block_weights(366, np.random.default_rng(5))
        self.assertEqual(w.sum(), 366)
        self.assertTrue((w >= 0).all())


if __name__ == '__main__':
    unittest.main()
