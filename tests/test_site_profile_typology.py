"""Check shape invariance, re-sampling, partition policy and null geometry."""
import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from site_profile_typology import (seasonal_profiles, choose_partition,
    fit_partition, align_labels, gaussian_reference, block_weights, minimum_size)


class TypologyChecks(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(18)
        self.quarters = np.repeat(np.arange(1, 5), 12)
        self.cube = self.rng.uniform(.01, .7, size=(48, 8, 11))

    def test_shape_is_invariant_to_site_and_season_scale(self):
        f, s, _, _ = seasonal_profiles(self.cube, self.quarters)
        factor = np.linspace(.2, 3., 8)[None, :, None]
        seasonal = np.array([.7, 1.4, 2., .9])[self.quarters - 1, None, None]
        altered, _, _, _ = seasonal_profiles(self.cube * factor * seasonal, self.quarters)
        np.testing.assert_allclose(f, altered, atol=1e-14)
        np.testing.assert_allclose(s.sum(axis=2), 1)
        self.assertEqual(f.shape, (8, 44))

    def test_weighted_profiles_match_explicit_repeated_dates(self):
        weights = self.rng.integers(1, 4, size=48)
        got = seasonal_profiles(self.cube, self.quarters, weights=weights)
        index = np.repeat(np.arange(48), weights)
        want = seasonal_profiles(self.cube[index], self.quarters[index])
        for a, b in zip(got, want):
            np.testing.assert_allclose(a, b, atol=1e-14)

    def test_weather_condition_selects_whole_daily_curve(self):
        mask = np.ones((48, 8), dtype=bool)
        mask[::2] = False
        _, _, raw, counts = seasonal_profiles(self.cube, self.quarters, condition=mask)
        expected = np.stack([self.cube[(self.quarters == q) & (np.arange(48) % 2 == 1)].mean(axis=0)
                             for q in range(1, 5)], axis=1)
        np.testing.assert_allclose(raw, expected)
        np.testing.assert_array_equal(counts, 6)

    def test_empty_quarter_and_zero_generation_are_not_filled(self):
        weights = np.ones(48)
        weights[self.quarters == 2] = 0
        with self.assertRaises(ValueError):
            seasonal_profiles(self.cube, self.quarters, weights=weights)
        altered = self.cube.copy()
        altered[:, 0] = 0
        with self.assertRaises(ValueError):
            seasonal_profiles(altered, self.quarters)

    def test_recovers_two_distinct_shapes_without_label_meaning(self):
        hours = np.arange(7, 18)
        curves = np.stack([np.exp(-(hours - 10) ** 2 / 4), np.exp(-(hours - 14) ** 2 / 4)])
        curves /= curves.sum(axis=1, keepdims=True)
        profiles = np.repeat(np.repeat(curves[:, None, :], 4, axis=1), 20, axis=0)
        profiles += self.rng.uniform(0, .0001, size=profiles.shape)
        profiles /= profiles.sum(axis=2, keepdims=True)
        x = profiles.reshape(40, 44) / 2
        with threadpool_limits(limits=1):
            fit = fit_partition(x, 2, self.rng, 'synthetic_test')
        self.assertTrue(fit['admissible'])
        self.assertAlmostEqual(adjusted_rand_score(np.repeat([0, 1], 20), fit['labels']), 1)
        self.assertGreater(fit['silhouette'], .99)

    def test_size_gate_can_reject_high_silhouette_outlier_split(self):
        outlier = {'k': 2, 'silhouette': .9, 'admissible': False}
        balanced = {'k': 3, 'silhouette': .5, 'admissible': True}
        self.assertEqual(choose_partition([outlier, balanced])['k'], 3)
        self.assertIsNone(choose_partition([outlier]))
        self.assertEqual(choose_partition([outlier], admissible=False)['k'], 2)
        self.assertEqual(minimum_size(80), 8)

    def test_labels_are_matched_by_permutation(self):
        a = np.repeat([0, 1, 2], 10)
        b = np.array([2, 0, 1])[a]
        np.testing.assert_array_equal(align_labels(a, b, 3), a)
        self.assertAlmostEqual(adjusted_rand_score(a, b), 1)

    def test_gaussian_reference_preserves_affine_season_constraints(self):
        f, _, _, _ = seasonal_profiles(self.cube, self.quarters)
        draw = gaussian_reference(f, self.rng)
        self.assertEqual(draw.shape, f.shape)
        np.testing.assert_allclose(draw.reshape(8, 4, 11).sum(axis=2), .5, atol=1e-12)

    def test_block_sample_conserves_number_of_dates(self):
        weights = block_weights(366, self.rng)
        self.assertEqual(weights.shape, (366,))
        self.assertEqual(weights.sum(), 366)
        self.assertTrue((weights >= 0).all())


if __name__ == '__main__':
    unittest.main()
