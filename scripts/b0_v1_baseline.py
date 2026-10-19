"""Fit B0 V1 only. Models, predictions and numeric outputs stay in ignored data/."""
from __future__ import annotations

import ast
import gc
import json
import platform
import sys
import sysconfig
import time
import unittest
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow

from s0_validation_check import (SOURCE, ROOT, INPUTS, baseline_contract, digest,
                                 load_forecasts, log, monthly, now, write_json)
from validation_core import (KEY, ROUNDS, WINDOWS, grid, group_totals, join_forecasts,
                             local_score, metadata, require, times, truth_matrix,
                             unique, weights)

OUT = ROOT / "data/validation/2026-10-19_B0_V1"
REPORT = ROOT / "reports/2026-10-19_20_B0_V1_기준선_학습과_평가.md"


def model_parameters():
    tree = ast.parse((SOURCE / "baseline.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "LGBMRegressor"]
    require(len(calls) == 1, "Baseline estimator ambiguous")
    params = {k.arg: ast.literal_eval(k.value) for k in calls[0].keywords}
    require(params == dict(n_estimators=600, learning_rate=0.06, num_leaves=127,
                           min_child_samples=200, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, verbose=-1, n_jobs=-1), "Baseline settings changed")
    return params


def add_features(d, spec):
    ts = pd.DatetimeIndex(d.ts)
    d["elev"], d["azim"], d["cosz"] = spec["solar"](ts, d.lat.to_numpy(), d.lon.to_numpy())
    doy = ts.dayofyear.to_numpy()
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype("float32")
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype("float32")
    d["hour"] = ts.hour.to_numpy().astype("int8")
    d["cap"] = d.capacity_kw.astype("float32")
    x = d[spec["FEATS"]].astype("float32")
    require(not np.isinf(x.to_numpy()).any(), "Infinite model input")
    require(x[spec["FEATS"][15:]].notna().all().all(), "Missing derived feature")
    return d, x


def feature_frame(base, fa, fb, r, spec):
    axis = base.ts.unique()
    return add_features(join_forecasts(base, fa.loc[fa.ts.isin(axis)], fb.loc[fb.ts.isin(axis)], r), spec)


def target_values(d):
    return (d.energy / d.cap).clip(0, 1.2).astype("float32")


def prediction_values(d, cf):
    require(len(cf) == len(d) and np.isfinite(cf).all(), "Invalid model output")
    result = np.clip(np.where(d.elev.to_numpy() > 0, cf, 0.0), 0, None) * d.cap.to_numpy()
    require(np.isfinite(result).all() and (result >= 0).all(), "Invalid kWh output")
    return result


def check_feature_parity(base, fa, fb, spec):
    """Compare actual rows with the original pure feature function, including NaNs."""
    tree = ast.parse((SOURCE / "baseline.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "features")
    namespace = {"np": np, "pd": pd, "solar": spec["solar"]}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<baseline-features>", "exec"), namespace)
    sample = pd.concat([base.head(128), base.tail(128)]).drop_duplicates(KEY)
    # Add rows whose weather is missing to exercise native NaN handling.
    for r in ROUNDS:
        joined, _ = feature_frame(base, fa, fb, r, spec)
        missing = joined.loc[joined[spec["FEATS"][:15]].isna().any(axis=1), KEY].head(32)
        part = pd.concat([sample, base.merge(missing, on=KEY, how="inner")]).drop_duplicates(KEY)
        actual, x = feature_frame(part, fa, fb, r, spec)
        original = namespace["features"](part[["site_id", "ts", "energy"]],
                        base.drop_duplicates("site_id")[["site_id", "current_group", "capacity_kw", "lat", "lon", "wa_station", "wb_station"]],
                        fa, fb, None, r)
        pd.testing.assert_frame_equal(x.reset_index(drop=True), original[spec["FEATS"]].astype("float32").reset_index(drop=True))
        np.testing.assert_array_equal(target_values(actual), target_values(original))
        del joined, original, x
    return True


def score_samples(truth, prediction, groups, price, fixed_ids):
    rows = []
    start = truth.index - pd.Timedelta(hours=1)
    for name, ids in (("fixed_148", fixed_ids), ("common_253", truth.columns.tolist())):
        a, y = truth[ids], prediction[ids]
        _, complete = group_totals(a, groups)
        masks = [("all", np.ones(len(start), dtype=bool))]
        masks += [(f"Q{q}", start.quarter == q) for q in range(1, 5)]
        masks += [(f"M{m:02}", start.month == m) for m in range(1, 13)]
        masks += [(f"k={k}", weights(truth.index) == k) for k in (1, 3)]
        for period, mask in masks:
            axis = truth.index[complete.to_numpy() & mask]
            if not len(axis):
                continue
            score = local_score(a, y, groups, price, axis)
            # Independent direct group accumulation verifies the public scoring path.
            p = price.set_index("ts").loc[axis].smp.to_numpy()
            direct = 0.0
            for group in ("G1", "G2"):
                members = [i for i in ids if groups[i] == group]
                err = y.loc[axis, members].to_numpy().sum(axis=1) - a.loc[axis, members].to_numpy().sum(axis=1)
                direct += float(np.sum(np.abs(err) * p * weights(axis)))
            require(np.isclose(direct, score["W_local"], rtol=1e-12, atol=1e-5), "Independent W mismatch")
            rows.append(dict(sample=name, period=period, first_ts=str(axis.min()), last_ts=str(axis.max()),
                             smp_sum=float(p.sum()), zero_price_times=int((p == 0).sum()), **score))
    return rows


def individual_metrics(truth, prediction, sites, trained_ids, spec):
    axis = truth.index
    start = axis - pd.Timedelta(hours=1)
    cap = sites.set_index("site_id").capacity_kw.reindex(truth.columns).to_numpy()
    error = prediction.to_numpy() - truth.to_numpy()
    valid = truth.notna().to_numpy()
    elev = np.stack([spec["solar"](axis, np.full(len(axis), row.lat), np.full(len(axis), row.lon))[0]
                     for row in sites.itertuples()], axis=1)
    masks = [("all", valid)]
    masks += [(f"Q{q}", valid & (start.quarter == q)[:, None]) for q in range(1, 5)]
    masks += [(f"M{m:02}", valid & (start.month == m)[:, None]) for m in range(1, 13)]
    masks += [(f"k={k}", valid & (weights(axis) == k)[:, None]) for k in (1, 3)]
    masks += [("solar_day", valid & (elev > 0)), ("solar_night", valid & (elev <= 0))]
    trained = truth.columns.isin(trained_ids)
    masks += [("has_train", valid & trained[None, :]), ("no_train", valid & ~trained[None, :])]
    rows = []
    cf_error = error / cap[None, :]
    for label, mask in masks:
        require(mask.any(), "Empty diagnostic mask")
        rows.append(dict(scope=label, rows=int(mask.sum()), sites=int(mask.any(axis=0).sum()),
                         cf_mae=float(np.abs(cf_error[mask]).mean()),
                         kwh_mae=float(np.abs(error[mask]).mean()),
                         bias_kwh=float(error[mask].mean()), cf_bias=float(cf_error[mask].mean())))
    per_site = pd.DataFrame({"site_id": truth.columns, "observed_rows": valid.sum(axis=0),
                            "has_train": trained, "cf_mae": np.nanmean(np.abs(cf_error), axis=0),
                            "kwh_mae": np.nanmean(np.abs(error), axis=0), "bias_kwh": np.nanmean(error, axis=0)})
    per_site.to_csv(OUT / "individual_sites.csv", index=False, encoding="utf-8-sig")
    return rows


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    require(sys.version_info[:2] == (3, 13) and not sysconfig.get_config_var("Py_GIL_DISABLED"), "Regular CPython 3.13 required")
    spec, params = baseline_contract(), model_parameters()
    runtime = dict(executable=sys.executable, python=platform.python_version(), lightgbm=lgb.__version__,
                   numpy=np.__version__, pandas=pd.__version__, pyarrow=pyarrow.__version__, free_threaded=False)
    log(f"Runtime: {runtime}; params={params}; random_state={lgb.LGBMRegressor(**params).random_state}")
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*validation*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    require(result.wasSuccessful(), "Validation tests failed")
    if "--preflight" in sys.argv:
        log(f"Preflight passed; {result.testsRun} tests; no fit/predict")
        return
    require(not (OUT / "summary.json").exists(), "Completed run exists; no automatic refit")
    OUT.mkdir(parents=True, exist_ok=True)
    started, t0 = now(), time.perf_counter()
    files = [Path(__file__), ROOT / "scripts/validation_core.py", ROOT / "scripts/s0_validation_check.py",
             ROOT / "tests/test_validation_core.py", ROOT / "tests/test_b0_validation.py"]
    code_hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
    archive = OUT / "executed_code"
    archive.mkdir(exist_ok=True)
    for p in files:
        (archive / p.name).write_bytes(p.read_bytes())
    before = {f: digest(SOURCE / f) for f in INPUTS}
    eda_hash = digest(ROOT / "reports/EDA_진행계획.md")
    s0 = json.loads((ROOT / "data/validation/2026-10-19_S0/summary.json").read_text(encoding="utf-8"))
    require(before == s0["source_sha256"] and eda_hash == s0["eda_plan_sha256"], "S0 input/EDA drift")
    sites = pd.read_csv(SOURCE / "sites.csv", encoding="utf-8-sig")
    capacities = metadata(sites)
    gen = pd.read_parquet(SOURCE / "train/generation.parquet")
    gen["ts"] = times(gen.ts)
    unique(gen, KEY)
    window = WINDOWS[0]
    require(window.name == "V1", "Only V1 allowed")
    axis = window.axis()
    truth = truth_matrix(gen, sites.site_id, axis)
    fixed_ids = truth.columns[truth.notna().all(axis=0)].tolist()
    require(fixed_ids == s0["fixed_148_sites"], "Diagnostic cohort drift")
    train = window.training(gen)
    require(len(train) == 1442111, "S0 train row mismatch")
    trained_ids = train.site_id.unique().tolist()
    base = train.merge(sites, on="site_id", how="left", validate="many_to_one", sort=False)
    price = pd.read_csv(SOURCE / "train/smp.csv", encoding="utf-8-sig")
    price["ts"] = times(price.ts)
    unique(price, ["ts"])
    require(np.isfinite(price.set_index("ts").reindex(axis).smp).all(), "V1 prices missing")
    del gen, train
    fa, fb, nonhourly = load_forecasts("train", spec)
    require(check_feature_parity(base, fa, fb, spec), "Feature parity failed")
    log("Original baseline features and target parity passed for both rounds, including missing weather")
    groups = sites.set_index("site_id").current_group.map({"A": "G1", "B": "G2"})
    predictions, fits = {}, []
    for r in ROUNDS:
        d, x = feature_frame(base, fa, fb, r, spec)
        day = d.elev.gt(0)
        y = target_values(d.loc[day])
        x = x.loc[day]
        require(len(x) == 719235 and len(x.columns) == 24 and "wa_ground_press" in x, "B0 training contract drift")
        fit_info = dict(round=r, train_rows=len(base), daylight_rows=len(x),
                        daylight_rows_any_weather_nan=int(x[spec["FEATS"][:15]].isna().any(axis=1).sum()),
                        clipped_daylight_targets=int(((d.loc[day].energy / d.loc[day].cap) != y).sum()),
                        train_ts_min=str(base.ts.min()), train_ts_max=str(base.ts.max()))
        # Count true clipping before float32 conversion, not precision conversion.
        raw_y = d.loc[day].energy / d.loc[day].cap
        fit_info["clipped_daylight_targets"] = int(((raw_y < 0) | (raw_y > 1.2)).sum())
        log(f"B0 V1 round={r}: fitting {x.shape}")
        fit_t0 = time.perf_counter()
        model = lgb.LGBMRegressor(**params)
        model.fit(x, y)
        fit_info.update(fit_seconds=time.perf_counter() - fit_t0, trees=model.booster_.num_trees(),
                        estimator_params=model.get_params(), booster_params=model.booster_.params)
        require(fit_info["trees"] == 600, "Unexpected tree count")
        path = OUT / f"model_round_{r}.txt"
        # Native LightGBM file IO cannot write this Windows non-ASCII path.
        path.write_text(model.booster_.model_to_string(), encoding="utf-8")
        fit_info["model_sha256"] = digest(path)
        del d, x, y, raw_y
        gc.collect()
        log(f"B0 V1 round={r}: fit complete; predicting monthly full grids")
        pieces = []
        for month_axis in monthly(axis):
            d, x = feature_frame(grid(sites, month_axis), fa, fb, r, spec)
            d["kwh"] = prediction_values(d, model.predict(x))
            pieces.append(d[KEY + ["kwh"]].copy())
            log(f"round={r}: {month_axis[0]}..{month_axis[-1]}, rows={len(d):,}")
        flat = pd.concat(pieces, ignore_index=True)
        unique(flat, KEY)
        require(len(flat) == len(axis) * 253, "Prediction grid lost rows")
        flat.to_parquet(OUT / f"predictions_round_{r}.parquet", index=False, compression="zstd")
        predictions[r] = flat.pivot(index="ts", columns="site_id", values="kwh").reindex(index=axis, columns=sites.site_id)
        require(np.isfinite(predictions[r].to_numpy()).all(), "Incomplete prediction matrix")
        fits.append(fit_info)
        del model, pieces, flat, d, x
        gc.collect()
    ensemble = (predictions[10] + predictions[17]) / 2
    ensemble.rename_axis(index="ts", columns="site_id").stack().rename("kwh").reset_index().to_parquet(
        OUT / "predictions_mean.parquet", index=False, compression="zstd")
    grouped = score_samples(truth, ensemble, groups, price, fixed_ids)
    individual = individual_metrics(truth, ensemble, sites, trained_ids, spec)
    pd.DataFrame(grouped).to_csv(OUT / "group_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(individual).to_csv(OUT / "individual_scores.csv", index=False, encoding="utf-8-sig")
    for sample in ("fixed_148", "common_253"):
        overall = next(s for s in grouped if s["sample"] == sample and s["period"] == "all")
        quarters = [s for s in grouped if s["sample"] == sample and s["period"].startswith("Q")]
        months = [s for s in grouped if s["sample"] == sample and s["period"].startswith("M")]
        for parts in (quarters, months):
            require(sum(p["times"] for p in parts) == overall["times"], "Sample partition mismatch")
            for key in ("W_local", "R_local", "S_local", "price_k_sum"):
                require(np.isclose(sum(p[key] for p in parts), overall[key], rtol=1e-12), "Metric partition mismatch")
    require(before == {f: digest(SOURCE / f) for f in INPUTS}, "Source changed")
    require(eda_hash == digest(ROOT / "reports/EDA_진행계획.md"), "EDA state changed")
    require(code_hashes == {str(p.relative_to(ROOT)): digest(p) for p in files}, "Code changed during run")
    failure_path = OUT / "failed_attempts.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8-sig")) if failure_path.exists() else None
    summary = dict(experiment="B0_V1", status="passed", started_at=started, finished_at=now(),
                   wall_seconds=time.perf_counter() - t0, runtime=runtime, source_sha256=before,
                   code_sha256=code_hashes, eda_plan_sha256=eda_hash, source_unchanged=True,
                   eda_state_unchanged=True, model_fit_count=2,
                   total_fit_count=2 + (failure["completed_fits"] if failure else 0), failed_attempt=failure,
                   B1_executed=False, V2_executed=False,
                   final_evaluation_executed=False, official_submission=False, synthetic_tests=result.testsRun,
                   feature_parity=True, independent_score_check=True, partition_checks=True,
                   model_features=spec["FEATS"], model_parameters=params, random_state=None,
                   seed_note="Unchanged baseline random_state=None; LightGBM default seeds, no seed search; exact bitwise reproducibility is not claimed.",
                   training_window=vars(window), train_sites=len(trained_ids), zero_train_sites=253-len(trained_ids),
                   prediction_times=len(axis), prediction_rows_per_round=len(axis)*253,
                   observed_truth_rows=int(truth.notna().to_numpy().sum()), fixed_148_sites=fixed_ids,
                   capacity_kw={k: float(v) for k, v in capacities.items()}, nonhourly=nonhourly,
                   fits=fits, group_scores=grouped, individual_scores=individual)
    write_json(OUT / "summary.json", summary)
    log(f"B0 V1 passed; 2 successful-run fits, total fits={summary['total_fit_count']}; ensemble scored; B1/V2/final evaluation not executed")


if __name__ == "__main__":
    main()
