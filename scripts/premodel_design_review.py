"""P-007 synthetic method checks. No data reads, model fitting or inference.

The LightGBM private adapter calls are pinned inspection probes, not a proposed
production API. Metadata roundtrip is NOT a trained-model roundtrip test.
"""
from __future__ import annotations

import ast
from contextlib import ExitStack
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import sysconfig
import unittest
from unittest.mock import patch

import lightgbm as lgb
from lightgbm.basic import _data_from_pandas, _dump_pandas_categorical, _load_pandas_categorical
import numpy as np
import pandas as pd

from b0_v1_baseline import prediction_values, target_values
from s0_validation_check import baseline_contract
from validation_core import KST, Window, clock, local_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/validation/2026-10-22_P007_design_review"
REPORT = ROOT / "reports/2026-10-22_02_모델링전_문제정의와_방법검토.md"
PLAN = ROOT / "reports/성능개선_진행계획.md"
RESULTS = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def windows_from_source():
    """Extract only literal development/final window definitions; no data access."""
    found = {}
    for name in ("b2_joint_compare.py", "validation_core.py", "b5_rolling_compare.py",
                 "b3_final_design_check.py"):
        tree = ast.parse((ROOT / "scripts" / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Window":
                args = [ast.literal_eval(arg) for arg in node.args]
                if args[0] in ("T0", "V1", "R3", "R4", "B3_final"):
                    found[args[0]] = Window(*args)
    if set(found) != {"T0", "V1", "R3", "R4", "B3_final"}:
        raise ValueError("Unexpected window definitions")
    return found


def synthetic_frame(ids, categories):
    spec = baseline_contract()  # Reads baseline source literals, not dataset rows.
    columns = ([f"{x}_r10" for x in spec["FEATS"][:15]]
               + [f"{x}_r17" for x in spec["FEATS"][:15]] + spec["FEATS"][15:])
    x = pd.DataFrame(np.arange(len(ids) * 39, dtype="float32").reshape(len(ids), 39), columns=columns)
    x.iloc[0, 0] = np.nan
    x["site_id_category"] = pd.Categorical(ids, categories=categories, ordered=False)
    return x


def adapter(frame, categories=None):
    return _data_from_pandas(frame, "auto", "auto", categories)


class MethodChecks(unittest.TestCase):
    def test_01_individual_loss_can_improve_while_W_worsens(self):
        axis = pd.DatetimeIndex(["2024-01-01 11:00"], tz=KST)
        truth = pd.DataFrame([[50., 50., 50., 50.]], index=axis, columns=list("abcd"))
        groups = pd.Series(["G1", "G1", "G2", "G2"], index=truth.columns)
        price = pd.DataFrame({"ts": axis, "smp": [2.]})
        before = truth + [10., -9., 0., 0.]
        after = truth + [2., 2., 0., 0.]
        old = local_score(truth, before, groups, price, axis)["W_local"]
        new = local_score(truth, after, groups, price, axis)["W_local"]
        row_old = float((before - truth).abs().to_numpy().sum())
        row_new = float((after - truth).abs().to_numpy().sum())
        self.assertEqual((row_old, row_new, old, new), (19., 4., 6., 24.))
        # At equal positive capacities the CF-L1 ordering is the same.
        # q*sum(abs(kWh errors)) upper-bounds W but need not rank candidates alike.
        self.assertLessEqual(old, 6 * row_old)
        self.assertLessEqual(new, 6 * row_new)
        RESULTS["01"] = dict(individual_abs_sum=[row_old, row_new], W=[old, new],
                             weighted_row_L1=[6 * row_old, 6 * row_new])

    def test_02_subcohort_W_is_not_additive(self):
        residual = np.array([10., -9., 0., 0.])
        within = abs(residual[:2].sum()) + abs(residual[2:].sum())
        split = np.abs(residual).sum()
        self.assertEqual((within, split), (1., 19.))
        cross = abs(10.) + abs(-10.)
        self.assertEqual(cross, 20.)
        self.assertEqual(abs(10. - 10.), 0.)
        RESULTS["02"] = dict(full_unweighted_group_error=float(within),
                             separate_subcohort_abs_sum=float(split),
                             opposite_errors_across_groups=float(cross))

    def test_03_sum_of_marginal_medians_is_not_median_of_sum(self):
        outcomes = np.array([[0., 0.], [0., 100.], [100., 0.]])
        individual = float(np.median(outcomes, axis=0).sum())
        aggregate = float(np.median(outcomes.sum(axis=1)))
        self.assertEqual((individual, aggregate), (0., 100.))
        loss0 = float(np.abs(outcomes.sum(axis=1) - individual).sum())
        loss1 = float(np.abs(outcomes.sum(axis=1) - aggregate).sum())
        self.assertEqual((loss0, loss1), (200., 100.))
        RESULTS["03"] = dict(sum_medians=individual, median_sum=aggregate,
                             aggregate_absolute_loss=[loss0, loss1])

    def test_04_target_clip_and_output_rules_differ(self):
        d = pd.DataFrame({"energy": [-10., 150., 50.], "cap": np.full(3, 100., dtype="float32"),
                          "elev": [10., 10., -1.]})
        original = d.copy(deep=True)
        target = target_values(d)
        np.testing.assert_allclose(target, [0., 1.2, .5])
        output = prediction_values(d, np.array([-0.1, 1.5, .5]))
        np.testing.assert_array_equal(output, [0., 150., 0.])
        pd.testing.assert_frame_equal(d, original)
        RESULTS["04"] = dict(raw_energy=d.energy.tolist(), target_cf=target.tolist(),
                             synthetic_output=output.tolist(), input_unchanged=True)

    def test_05_overlap_and_fit_age_are_not_forecast_lead(self):
        windows = windows_from_source()
        v1, r3, r4 = [windows[x].axis() for x in ("V1", "R3", "R4")]
        self.assertEqual((len(v1.intersection(r3)), len(v1.intersection(r4)),
                          len(r3.intersection(r4))), (2208, 2208, 0))
        details = {}
        for name, w in windows.items():
            axis = w.axis()
            c = clock(axis)
            lead = (c.ts - c.prediction_at).dt.total_seconds() / 3600
            age = (axis - pd.Timestamp(w.cutoff, tz=KST)).total_seconds() / 3600
            self.assertEqual((float(lead.min()), float(lead.max())), (8., 31.))
            self.assertEqual(float(age.min()), 25.)
            self.assertEqual(c.prediction_at.min() - pd.Timestamp(w.cutoff, tz=KST), pd.Timedelta(hours=17))
            details[name] = dict(hours=len(axis), forecast_lead_hours=[8, 31],
                                 time_since_training_cutoff_hours=[float(age.min()), float(age.max())])
        RESULTS["05"] = dict(V1_R3_overlap=2208, V1_R4_overlap=2208, R3_R4_overlap=0, windows=details)

    def test_06_category_dictionary_excludes_future_and_night_only_ids(self):
        w = windows_from_source()["T0"]
        c = pd.Timestamp(w.cutoff, tz=KST)
        d = pd.DataFrame({"site_id": ["seen", "future_only", "night_only"],
                          "ts": [c, c + pd.Timedelta(hours=1), c - pd.Timedelta(hours=1)],
                          "elev": [10., 10., -1.]})
        eligible = w.training(d)
        categories = sorted(eligible.loc[eligible.elev.gt(0), "site_id"].unique())
        self.assertEqual(categories, ["seen"])
        RESULTS["06"] = dict(categories=categories, future_and_night_only_excluded=True)

    def test_07_category_batch_and_order_alignment(self):
        train = synthetic_frame(["b", "a", "c"], ["a", "b", "c"])
        _, _, cat_cols, dictionary = adapter(train)
        self.assertEqual(cat_cols, ["site_id_category"])
        valid = synthetic_frame(["c", "a", "b"], ["c", "b", "a"])
        full = adapter(valid, dictionary)[0]
        np.testing.assert_array_equal(full[:, -1], [2, 0, 1])
        chunks = np.concatenate([adapter(valid.iloc[:1], dictionary)[0],
                                 adapter(valid.iloc[1:], dictionary)[0]])
        np.testing.assert_array_equal(full, chunks)
        shuffled = adapter(valid.iloc[[2, 0, 1]], dictionary)[0]
        np.testing.assert_array_equal(full, shuffled[[1, 2, 0]])
        RESULTS["07"] = dict(aligned_codes=full[:, -1].tolist(), batch_and_order_invariant=True)

    def test_08_mixed_category_preserves_original_39_features(self):
        x = synthetic_frame(["a", "b", "c"], ["a", "b", "c"])
        before = x.copy(deep=True)
        data, names, cat_cols, _ = adapter(x)
        pd.testing.assert_frame_equal(before, x)
        self.assertTrue(x.iloc[:, :39].dtypes.eq("float32").all())
        self.assertIsInstance(x.dtypes.iloc[-1], pd.CategoricalDtype)
        np.testing.assert_array_equal(data[:, :39], x.iloc[:, :39].to_numpy())
        self.assertEqual((len(names), cat_cols), (40, ["site_id_category"]))
        # The old all-float32 assertion would reject this intended mixed schema.
        self.assertFalse(x.dtypes.eq("float32").all())
        RESULTS["08"] = dict(original_columns=39, total_columns=40, original_values_unchanged=True,
                             original_dtype="float32", category_dtype=str(x.dtypes.iloc[-1]),
                             adapter_dtype=str(data.dtype))

    def test_09_unknown_becomes_missing_but_plain_codes_are_numeric(self):
        dictionary = [["a", "b", "c"]]
        x = synthetic_frame(["a", "new_site", "b"], ["new_site", "b", "a"])
        data, _, cat_cols, _ = adapter(x, dictionary)
        self.assertEqual(cat_cols, ["site_id_category"])
        self.assertTrue(np.isnan(data[1, -1]))
        np.testing.assert_array_equal(data[[0, 2], -1], [0, 1])
        numeric = pd.DataFrame({"site_id_category": np.array([0, 1, 2], dtype="int32")})
        self.assertEqual(adapter(numeric)[2], [])
        RESULTS["09"] = dict(unknown_is_nan=True, inferred_categorical_columns=cat_cols,
                             plain_integer_auto_categorical_columns=[],
                             B3_prediction_fallback_tested=False)

    def test_10_category_metadata_roundtrip_only(self):
        categories = [["가상_A", "가상_B"]]
        serialized = _dump_pandas_categorical(categories)
        restored = _load_pandas_categorical(model_str=serialized)
        self.assertEqual(restored, categories)
        self.assertEqual(json.loads(json.dumps(categories, ensure_ascii=False)), categories)
        RESULTS["10"] = dict(metadata_roundtrip=True, trained_model_reload_tested=False)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.version_info[:2] != (3, 13) or sysconfig.get_config_var("Py_GIL_DISABLED"):
        raise RuntimeError("Regular CPython 3.13 required")
    if lgb.__version__ != "4.6.0":
        raise RuntimeError("Re-review private adapter probes for this LightGBM version")
    OUT.mkdir(parents=True, exist_ok=True)
    pre = OUT / "preregistered_report.md"
    if not pre.exists():
        pre.write_bytes(REPORT.read_bytes())
        (OUT / "preregistered_plan.md").write_bytes(PLAN.read_bytes())
    paths = [Path(__file__), ROOT / "scripts/validation_core.py", ROOT / "scripts/b0_v1_baseline.py",
             ROOT / "scripts/b2_joint_compare.py", ROOT / "scripts/b5_rolling_compare.py",
             ROOT / "scripts/b3_final_design_check.py", ROOT / "scripts/s0_validation_check.py"]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    blocked = []

    def forbidden(*args, **kwargs):
        blocked.append("forbidden_runtime_call")
        raise AssertionError("P-007 forbids dataset row reads, fitting and model inference")

    with ExitStack() as stack:
        for obj, attr in [(pd, "read_csv"), (pd, "read_parquet"), (lgb, "train"), (lgb, "cv"),
                          (lgb.LGBMRegressor, "fit"), (lgb.LGBMRegressor, "predict"),
                          (lgb.Booster, "__init__"), (lgb.Booster, "predict")]:
            stack.enter_context(patch.object(obj, attr, side_effect=forbidden))
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(MethodChecks)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    unchanged = hashes == {str(p.relative_to(ROOT)): sha(p) for p in paths}
    ok = result.wasSuccessful() and result.testsRun == 10 and len(RESULTS) == 10 and not blocked and unchanged
    value = dict(status="passed" if ok else "failed", tests_run=result.testsRun,
                 failures=len(result.failures), errors=len(result.errors),
                 executed_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                 runtime=dict(python=sys.version, executable=sys.executable, lightgbm=lgb.__version__,
                              pandas=pd.__version__, numpy=np.__version__),
                 scope="synthetic methods and adapter checks only", model_fits=0, model_inference_calls=0,
                 real_data_row_reads=0, forbidden_calls=blocked, code_unchanged=unchanged,
                 source_sha256=hashes, lightgbm_basic_sha256=sha(Path(lgb.basic.__file__)),
                 preregistered_report_sha256=sha(pre),
                 preregistered_plan_sha256=sha(OUT / "preregistered_plan.md"), results=RESULTS)
    (OUT / "summary.json").write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: value[k] for k in ("status", "tests_run", "model_fits", "real_data_row_reads")}, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
