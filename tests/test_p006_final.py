"""Final ID alignment and category contracts; never fit a model."""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from p006_final import assemble
from p006_site_category import categorical_features, categories_from_training, NEW


class FinalAlignment(unittest.TestCase):
    def parts(self):
        x = pd.DataFrame(np.arange(156, dtype="float32").reshape(4, 39), index=[8, 2, 7, 1])
        x.columns = [f"x{i}" for i in range(39)]
        x.iloc[0, 0] = np.nan
        y = pd.Series([.8, .2, .7, .1], index=x.index, dtype="float32", name="target")
        ids = pd.Series(["b", "a", "b", "c"], index=x.index, name="site_id")
        return x, y, ids

    def test_independent_part_orders_restore_id_target_and_numeric(self):
        x, y, i = self.parts()
        xx, yy, ii, keys = assemble([x.iloc[:2], x.iloc[2:]], [y.iloc[::-1]], [i.iloc[[2, 0]], i.iloc[[3, 1]]])
        self.assertEqual(keys.tolist(), [1, 2, 7, 8])
        pd.testing.assert_frame_equal(xx, x.sort_index().reset_index(drop=True))
        pd.testing.assert_series_equal(yy, y.sort_index().reset_index(drop=True))
        self.assertEqual(ii.tolist(), ["c", "a", "b", "b"])
        final = categorical_features(xx, ii, categories_from_training(ii))
        self.assertEqual(final[NEW].cat.codes.tolist(), [2, 0, 1, 1])
        self.assertEqual(final.shape, (4, 40))

    def test_missing_id_key_rejected(self):
        x, y, i = self.parts()
        with self.assertRaises(ValueError):
            assemble([x], [y], [i.iloc[:-1]])

    def test_duplicate_id_key_rejected(self):
        x, y, i = self.parts()
        with self.assertRaises(ValueError):
            assemble([x], [y], [i, i.iloc[:1]])

    def test_missing_target_and_bad_dtype_rejected(self):
        x, y, i = self.parts()
        with self.assertRaises(ValueError):
            assemble([x], [y.iloc[:-1]], [i])
        with self.assertRaises(ValueError):
            assemble([x.astype("float64")], [y], [i])

    def test_missing_site_value_rejected(self):
        x, y, i = self.parts()
        i.iloc[0] = None
        with self.assertRaises(ValueError):
            assemble([x], [y], [i])


if __name__ == "__main__":
    unittest.main(verbosity=2)
