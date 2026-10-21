"""Boundary checks for common support, profiles and unbalanced weighted effects."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from cohort_applicability import (block_weights, linear_design, masked_cross, masked_stats,
    monthly_eligibility, profile_metrics, solve, weather_caliper)


class CohortChecks(unittest.TestCase):
    def test_profile_is_scale_invariant_except_level(self):
        x = np.array([1, 2, 3, 4, 5, 5, 4, 3, 2, 1, .5])
        a, b = profile_metrics(x), profile_metrics(3 * x)
        np.testing.assert_allclose(a[:2], b[:2])
        self.assertAlmostEqual(b[2], 3 * a[2])
        self.assertIsNone(profile_metrics(np.zeros(11)))

    def test_monthly_threshold_does_not_borrow_another_month(self):
        months = np.repeat([1, 2], 20)
        complete = np.zeros((40, 2), dtype=bool)
        complete[:14, 0] = True
        complete[20:33, 0] = True
        complete[:, 1] = True
        result = monthly_eligibility(complete, months)
        self.assertTrue(result[:20, 0].all())
        self.assertFalse(result[20:, 0].any())
        self.assertTrue(result[:, 1].all())

    def test_calipers_are_joint_inclusive_and_wet_class_is_strict(self):
        target = np.array([[50., 50., 20., 3., 3., .1]])
        references = np.array([[[65., 65., 25., 5., 4.5, .1],
                                [65.001, 50., 20., 3., 3., .1],
                                [50., 50., 20., 3., 3., .10001],
                                [50., np.nan, 20., 3., 3., .1]]])
        np.testing.assert_array_equal(weather_caliper(target, references), [[True, False, False, False]])

    def test_linear_rain_features_keep_valid_dry_zero(self):
        w = np.array([[50., 60., 20., 2., 3., 0.], [50., 60., 20., 2., 3., 1.]])
        x = linear_design(w)
        np.testing.assert_allclose(x[0], [5, 6, 2, 2, 3, 0, 0])
        np.testing.assert_allclose(x[1, -2:], [1, np.log(2)])

    def direct(self, x, y, mask, months, weights):
        groups = np.repeat((months - 1) * np.prod(mask.shape[1:]), np.prod(mask.shape[1:]))
        groups += np.tile(np.arange(np.prod(mask.shape[1:])), len(months))
        take = mask.ravel() & (np.repeat(weights, np.prod(mask.shape[1:])) > 0)
        dummy = np.eye(int(groups.max()) + 1)[groups[take]]
        design = np.column_stack([x.reshape(-1, x.shape[-1])[take], dummy])
        sw = np.sqrt(np.repeat(weights, np.prod(mask.shape[1:]))[take])
        return np.linalg.lstsq(design * sw[:, None], y.ravel()[take] * sw, rcond=None)[0][:x.shape[-1]]

    def test_unbalanced_weighted_absorption_matches_full_dummy_ols(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(30, 2, 3, 3))
        y = rng.normal(size=(30, 2, 3)) + np.repeat([0., 4., -2.], 10)[:, None, None]
        mask = rng.uniform(size=y.shape) > .25
        months = np.repeat([1, 2, 3], 10)
        weights = rng.integers(0, 5, size=30)
        got = solve(*masked_cross(masked_stats(x, y, mask, months), weights))
        self.assertTrue(got['valid'])
        np.testing.assert_allclose(np.array(got['beta_pp']) / 100,
                                   self.direct(x, y, mask, months, weights), atol=1e-10)

    def test_empty_month_and_nan_outside_support_are_ignored(self):
        rng = np.random.default_rng(19)
        x = rng.normal(size=(30, 2, 2, 2))
        y = rng.normal(size=(30, 2, 2))
        mask = rng.uniform(size=y.shape) > .3
        x[~mask] = np.nan
        y[~mask] = np.nan
        months = np.repeat([1, 2, 3], 10)
        weights = np.r_[np.ones(10), np.zeros(10), np.ones(10)]
        got = solve(*masked_cross(masked_stats(x, y, mask, months), weights))
        self.assertTrue(got['valid'])
        np.testing.assert_allclose(np.array(got['beta_pp']) / 100,
                                   self.direct(x, y, mask, months, weights), atol=1e-10)

    def test_joint_block_draw_preserves_date_count_and_is_reproducible(self):
        a = block_weights(184, 14, np.random.default_rng(3))
        b = block_weights(184, 14, np.random.default_rng(3))
        self.assertEqual(a.sum(), 184)
        np.testing.assert_array_equal(a, b)

    def test_rank_deficiency_is_rejected(self):
        xx = np.ones((2, 2))
        self.assertFalse(solve(xx, np.array([1., 1.]), 10)['valid'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
