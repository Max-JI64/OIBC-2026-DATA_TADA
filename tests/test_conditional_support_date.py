"""Joint-support boundary and exact two-way weighted fixed-effect checks."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from conditional_support_date import cross, joint_mask, prepare, prune_edges, solve, block_weights


def dummy_cross(x, y, mask, sizes, months, weights, date_effect):
    d, s, h, p = x.shape
    z = np.concatenate([x, y[..., None]], axis=-1).reshape(-1, p + 1)
    cell = np.broadcast_to((months[:, None, None] - 1) * s * h + np.arange(s * h).reshape(1, s, h), mask.shape).ravel()
    nuisance = np.eye(int(cell.max()) + 1)[cell]
    if date_effect:
        date = np.broadcast_to(np.arange(d)[:, None, None] * h + np.arange(h)[None, None], mask.shape).ravel()
        nuisance = np.column_stack([nuisance, np.eye(d * h)[date]])
    w = (weights[:, None, None] * sizes[None, :, None] * mask).ravel()
    valid = w > 0
    a = nuisance[valid] * np.sqrt(w[valid, None])
    b = z[valid] * np.sqrt(w[valid, None])
    residual = b - a @ np.linalg.lstsq(a, b, rcond=None)[0]
    return residual.T @ residual


class SupportDateChecks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.x = rng.normal(size=(18, 4, 2, 3))
        self.y = rng.normal(size=(18, 4, 2))
        self.mask = rng.uniform(size=self.y.shape) > .25
        self.sizes = np.array([1, 2, 4, 3])
        self.months = np.repeat([1, 2, 3], 6)
        self.weights = rng.integers(0, 5, size=18)

    def test_weighted_calendar_cross_equals_explicit_dummy_ols(self):
        stats = prepare(self.x, self.y, self.mask, self.sizes, self.months)
        got = cross(stats, self.weights, False)
        expected = dummy_cross(self.x, self.y, self.mask, self.sizes, self.months, self.weights, False)
        np.testing.assert_allclose(got, expected, atol=1e-9)

    def test_weighted_two_way_cross_equals_explicit_dummy_ols(self):
        stats = prepare(self.x, self.y, self.mask, self.sizes, self.months)
        got = cross(stats, self.weights, True)
        expected = dummy_cross(self.x, self.y, self.mask, self.sizes, self.months, self.weights, True)
        np.testing.assert_allclose(got, expected, atol=1e-9)

    def test_empty_month_and_disconnected_graph_are_valid_projections(self):
        weights = self.weights.copy()
        weights[self.months == 2] = 0
        mask = self.mask.copy()
        mask[:3, 2:] = False
        mask[3:6, :2] = False
        for date in [False, True]:
            got = cross(prepare(self.x, self.y, mask, self.sizes, self.months), weights, date)
            expected = dummy_cross(self.x, self.y, mask, self.sizes, self.months, weights, date)
            np.testing.assert_allclose(got, expected, atol=1e-9)

    def test_station_means_preserve_weather_coefficients_with_site_effects(self):
        rng = np.random.default_rng(8)
        mapping = np.array([0, 1, 1, 2, 2, 2, 2, 3, 3, 3])
        ysite = self.y[:, mapping] + rng.normal(size=(18, len(mapping), 2)) * .1
        ysite += rng.normal(size=(1, len(mapping), 2))
        avg = np.stack([ysite[:, mapping == st].mean(axis=1) for st in range(4)], axis=1)
        for date in [False, True]:
            compact = cross(prepare(self.x, avg, self.mask, self.sizes, self.months), self.weights, date)
            expanded = dummy_cross(self.x[:, mapping], ysite, self.mask[:, mapping], np.ones(len(mapping)),
                                   self.months, self.weights, date)
            b1 = np.linalg.solve(compact[:-1, :-1], compact[:-1, -1])
            b2 = np.linalg.solve(expanded[:-1, :-1], expanded[:-1, -1])
            np.testing.assert_allclose(b1, b2, atol=1e-10)

    def test_joint_support_in_same_month_hour_uses_other_covariates(self):
        w = np.zeros((8, 1, 1, 6))
        w[:, 0, 0] = [50, 60, 20, 3, 3, 0]
        w[:4, 0, 0, 0] = 25
        w[4:, 0, 0, 0] = 75
        mask, cells = joint_mask(w, np.ones(8), (0, 25., 75., 10.))
        self.assertEqual(mask.sum(), 8)
        self.assertEqual(cells[0]['edges'], 16)
        w[4:, 0, 0, 4] += 1.001
        self.assertFalse(joint_mask(w, np.ones(8), (0, 25., 75., 10.))[0].any())

    def test_support_is_not_borrowed_from_other_months(self):
        w = np.tile(np.array([25., 60., 20., 3., 3., 0.]), (8, 1, 1, 1))
        w[4:, ..., 0] = 75
        mask, _ = joint_mask(w, np.repeat([1, 2], 4), (0, 25., 75., 10.))
        self.assertFalse(mask.any())

    def test_bipartite_pruning_removes_single_neighbor_cascade(self):
        edges = np.array([[1, 1, 0], [1, 0, 0], [0, 1, 1]], dtype=bool)
        self.assertFalse(prune_edges(edges).any())
        self.assertTrue(prune_edges(np.ones((3, 3), dtype=bool)).all())

    def test_unidentified_target_is_rejected_and_constant_control_is_omitted(self):
        matrix = np.diag([2., 0., 3., 5.])
        matrix[0, -1] = matrix[-1, 0] = .3
        self.assertFalse(solve(matrix, 1)['valid'])
        good = solve(matrix, 0)
        self.assertTrue(good['valid'])
        self.assertIsNone(good['beta_pp'][1])

    def test_block_draw_is_reproducible_and_has_fixed_date_count(self):
        a = block_weights(365, 14, np.random.default_rng(3))
        b = block_weights(365, 14, np.random.default_rng(3))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.sum(), 365)


if __name__ == '__main__':
    unittest.main(verbosity=2)
