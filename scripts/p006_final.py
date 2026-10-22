"""Frozen P-006 final input check, one fit, local outputs and saved verification."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import sys
import sysconfig
import time
import traceback

import lightgbm as lgb
import numpy as np
import pandas as pd

from b0_v1_baseline import target_values, prediction_values
from b2_joint_compare import frames, joint_features, runtime
from b3_final_design_check import FINAL
from b3_final_fit_predict import combine_training_parts, validate_site_grid, validate_outputs
from p006_site_category import (NEW, CAT_PARAMS, candidate_parameters, categories_from_training,
                                categorical_features, validate_model, intact as development_intact)
from s0_validation_check import ROOT, SOURCE, baseline_contract, digest, load_forecasts, monthly, now, log, write_json
from validation_core import KEY, clock, grid, metadata, require, times, unique, weights

OUT = ROOT / "data/final/2026-10-22_P006"
OLD = ROOT / "data/final/2026-10-20_B3"
DESIGN = ROOT / "data/validation/2026-10-20_B3_final_design"
DEV = ROOT / "data/validation/2026-10-22_P006_site_category"
BLOCK = ROOT / "data/validation/2026-10-22_P010_block_stability"
REPORTS = [ROOT / "reports" / name for name in (
    "2026-10-22_05_P009_최종입력_계약점검.md", "2026-10-22_07_P011_최종학습과_로컬출력검증.md")]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def array_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def assemble(xs, ys, ids):
    """Keep original observation keys until numeric x, target and ID all align."""
    x = pd.concat(xs).sort_index(kind="stable")
    y, i = pd.concat(ys), pd.concat(ids)
    require(x.index.is_unique and y.index.is_unique and i.index.is_unique, "Duplicate original row key")
    require(set(x.index) == set(y.index) == set(i.index), "Original row-key sets differ")
    y, i = y.reindex(x.index), i.reindex(x.index)
    require(i.notna().all(), "Missing original site ID")
    numeric, yy = combine_training_parts(xs, ys)
    pd.testing.assert_frame_equal(numeric, x.reset_index(drop=True))
    pd.testing.assert_series_equal(yy, y.reset_index(drop=True))
    return numeric, yy, i.reset_index(drop=True), x.index.to_numpy()


def code_paths():
    names = ["p006_final.py", "p006_site_category.py", "b3_final_fit_predict.py", "b3_final_design_check.py",
             "b0_v1_baseline.py", "b2_joint_compare.py", "b3_l1_compare.py", "h1_cloud_interaction.py",
             "forecast_residual_eda.py", "s0_validation_check.py", "validation_core.py"]
    return [ROOT / "scripts" / n for n in names] + [ROOT / "tests/test_p006_final.py"]


def snapshot():
    dev, ver = read(DEV / "summary.json"), read(DEV / "verification.json")
    require(dev["selection"] == ver["selection"] == "P006" and ver["status"] == "passed", "Development choice invalid")
    development_intact(read(DEV / "precheck.json"), live=False)
    old_pre, old_summary = read(OLD / "precheck.json"), read(OLD / "summary.json")
    source = {}
    for rel, h in old_pre["source_sha256"].items():
        require(digest(SOURCE / rel) == h, f"Final source drift: {rel}")
        source[str((SOURCE / rel).relative_to(ROOT))] = h
    require(old_summary["status"] == "passed", "Preserved B3 incomplete")
    for name, h in old_summary["output_sha256"].items():
        require(digest(OLD / name) == h, f"Preserved B3 output drift: {name}")
    for rel, h in read(DESIGN / "summary.json")["code_sha256"].items():
        require(digest(ROOT / rel) == h, "Final window/feature contract changed")
    paths = [p for folder in (OLD, DESIGN, DEV) for p in folder.rglob("*") if p.is_file()]
    return dict(source, **{str(p.relative_to(ROOT)): digest(p) for p in paths})


def training():
    design, old_fit = read(DESIGN / "summary.json"), read(OLD / "fit_summary.json")
    require(design["window"] == vars(FINAL), "Final time policy changed")
    spec = baseline_contract()
    sites = pd.read_csv(SOURCE / "sites.csv", encoding="utf-8-sig")
    metadata(sites)
    gen = pd.read_parquet(SOURCE / "train/generation.parquet")
    gen["ts"] = times(gen.ts)
    unique(gen, KEY)
    require(gen.site_id.isin(sites.site_id).all() and np.isfinite(gen.energy).all(), "Bad supplied truth")
    require(not gen.ts.isin(FINAL.axis()).any(), "Unexpected evaluation truth")
    t = FINAL.training(gen).reset_index(drop=True)
    t["_train_order"] = np.arange(len(t), dtype="int64")
    require(len(t) == design["train"]["rows"] and t.site_id.nunique() == 253, "Train rows/sites differ")
    require(str(t.ts.min()) == old_fit["train_ts_min"] and str(t.ts.max()) == old_fit["train_ts_max"], "Train bounds differ")
    require(t.ts.max() < clock(FINAL.axis()).prediction_at.min(), "Training availability leakage")
    del gen
    fa, fb, excluded = load_forecasts("train", spec)
    xs, ys, ids = [], [], []
    for axis in monthly(pd.DatetimeIndex(t.ts.unique()).sort_values()):
        base = t.loc[t.ts.isin(axis)].merge(sites, on="site_id", how="left", validate="many_to_one", sort=False)
        d, x10, d17, x17 = frames(base, fa, fb, spec)
        orders = d._train_order.to_numpy()
        pd.testing.assert_frame_equal(d[KEY].reset_index(drop=True), t.iloc[orders][KEY].reset_index(drop=True))
        x = joint_features(d, x10, d17, x17, spec)
        require(x.columns.tolist() == old_fit["features"], "Numeric feature schema differs")
        day = d.elev.gt(0)
        index = orders[day.to_numpy()]
        part, y, i = x.loc[day].copy(), target_values(d.loc[day]), d.loc[day, "site_id"].copy()
        part.index = y.index = i.index = index
        xs.append(part); ys.append(y); ids.append(i)
        log(f"P009 final inputs: {axis[0]:%Y-%m}")
        del d, base, x10, d17, x17, x, part, y, i
    numeric, y, ids, original_keys = assemble(xs, ys, ids)
    pd.testing.assert_series_equal(ids, t.iloc[original_keys].site_id.reset_index(drop=True))
    categories = categories_from_training(ids)
    require(set(categories) == set(sites.site_id) and len(categories) == 253, "Final dictionary lacks a target ID")
    require(len(numeric) == old_fit["daylight_rows"] == design["train"]["daylight_rows"], "Daylight count differs")
    require(int(numeric.isna().any(axis=1).sum()) == old_fit["any_feature_nan_rows"], "Final numeric NaN contract differs")
    meta = t.iloc[original_keys][["site_id", "ts"]].copy()
    counts = meta.groupby("site_id").size()
    require(int(counts.min()) == design["train"]["min_daylight_site_rows"] and
            int(counts.max()) == design["train"]["max_daylight_site_rows"], "Per-ID daylight bounds differ")
    xx = categorical_features(numeric, ids, categories)
    require(xx[NEW].notna().all() and xx.index.equals(y.index), "Category/target alignment differs")
    audit = dict(train_rows=len(t), daylight_rows=len(xx), train_ts_min=str(t.ts.min()), train_ts_max=str(t.ts.max()),
        features=numeric.columns.tolist(), new_features=xx.columns.tolist(), categories=categories,
        numeric_sha256=array_hash(numeric.to_numpy()), target_sha256=array_hash(y.to_numpy()),
        original_keys_sha256=array_hash(original_keys), id_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        category_codes_sha256=array_hash(xx[NEW].cat.codes.to_numpy()), any_feature_nan_rows=int(numeric.isna().any(axis=1).sum()),
        numeric_dtypes=list(map(str, numeric.dtypes)), target_dtype=str(y.dtype), unseen_target_ids=0,
        min_daylight_site_rows=int(counts.min()), max_daylight_site_rows=int(counts.max()), excluded_nonhourly=excluded,
        per_site=[dict(site_id=i, rows=len(z), months=int(z.ts.dt.strftime("%Y-%m").nunique()),
                       first_ts=str(z.ts.min()), last_ts=str(z.ts.max())) for i, z in meta.groupby("site_id", sort=True)])
    del fa, fb, t, xs, ys, ids, numeric, meta
    gc.collect()
    return xx, y, audit


def intact(pre):
    require(runtime() == pre["runtime"] and candidate_parameters() == pre["parameters"], "Runtime/settings drift")
    require(snapshot() == pre["protected_sha256"], "Protected artifact drift")
    for rel, h in pre["code_sha256"].items():
        require(digest(ROOT / rel) == digest(OUT / "executed_code" / Path(rel).name) == h, "Execution code drift")
    for rel, h in pre["scope_sha256"].items():
        require(digest(OUT / "preregistered" / Path(rel).name) == h, "Frozen scope drift")


def precheck():
    OUT.mkdir(parents=True, exist_ok=True)
    require(not (OUT / "precheck.json").exists(), "Final precheck already frozen")
    before = snapshot()
    x, y, audit = training()
    require(snapshot() == before, "Inputs changed during P009")
    (OUT / "executed_code").mkdir(exist_ok=True)
    (OUT / "preregistered").mkdir(exist_ok=True)
    for p in code_paths():
        (OUT / "executed_code" / p.name).write_bytes(p.read_bytes())
    for p in REPORTS:
        (OUT / "preregistered" / p.name).write_bytes(p.read_bytes())
    write_json(OUT / "precheck.json", dict(status="passed", at=now(), runtime=runtime(), parameters=candidate_parameters(),
        window=vars(FINAL), training=audit, protected_sha256=before, code_sha256={str(p.relative_to(ROOT)): digest(p) for p in code_paths()},
        scope_sha256={str(p.relative_to(ROOT)): digest(p) for p in REPORTS}, final_fit_budget=1, model_fit_count=0,
        model_prediction_count=0, performance_scores_computed=False))
    log("P009 passed: 253 known IDs, 39+1 features, aligned final inputs; fits/predictions=0")


def frame(axis, sites, fa, fb, spec, audit):
    d, x10, d17, x17 = frames(grid(sites, axis), fa, fb, spec)
    x = categorical_features(joint_features(d, x10, d17, x17, spec), d.site_id, audit["categories"])
    require(x[NEW].notna().all(), "Unseen final prediction ID")
    return d, x


def final_inputs():
    sites = pd.read_csv(SOURCE / "sites.csv", encoding="utf-8-sig")
    price = pd.read_csv(SOURCE / "smp.csv", encoding="utf-8-sig")
    price["ts"] = times(price.ts)
    unique(price, ["ts"])
    require(times(price.ts).equals(FINAL.axis()) and len(price) == 8760, "Final time axis drift")
    require(np.array_equal(price.k, weights(FINAL.axis())) and np.isfinite(price.smp).all() and price.smp.ge(0).all(), "Bad prices/k")
    return sites, price


def execute(resume=False):
    require(not (OUT / "summary.json").exists(), "Final execution complete; use --verify")
    pre = read(OUT / "precheck.json")
    intact(pre)
    block, block_verify = read(BLOCK / "summary.json"), read(BLOCK / "verification.json")
    require(block["status"] == block_verify["status"] == "passed" and
            block_verify["summary_sha256"] == digest(BLOCK / "summary.json"), "P010 must be checked first")
    started = now()
    model_path, journal = OUT / "model.txt", OUT / "fit_journal.json"
    if not resume:
        require(not journal.exists() and not model_path.exists(), "Fit already attempted; never refit automatically")
        x, y, audit = training()
        require(audit == pre["training"], "Final training contract drift")
        write_json(journal, dict(status="started", at=now(), fit_attempts=1))
        try:
            tick = time.perf_counter()
            estimator = lgb.LGBMRegressor(**candidate_parameters()).fit(x, y, categorical_feature=[NEW])
            elapsed = time.perf_counter() - tick
            text = estimator.booster_.model_to_string()
            model_path.write_text(text, encoding="utf-8")
            model = lgb.Booster(model_str=model_path.read_text(encoding="utf-8"))
            validate_model(model, audit, text)
            positions = np.unique(np.linspace(0, len(x)-1, 1024, dtype=int))
            np.testing.assert_allclose(estimator.booster_.predict(x.iloc[positions]), model.predict(x.iloc[positions]), rtol=1e-12, atol=0)
            write_json(OUT / "fit_summary.json", dict(status="passed", at=now(), model_fit_count=1, fit_attempts=1,
                training=audit, parameters=candidate_parameters(), fit_seconds=elapsed, model_sha256=digest(model_path),
                reload_training_probe_rows=len(positions)))
            write_json(journal, dict(status="saved", at=now(), fit_attempts=1, model_sha256=digest(model_path)))
            del estimator, x, y
        except BaseException:
            write_json(journal, dict(status="failed", at=now(), fit_attempts=1, error=traceback.format_exc()))
            raise
    fit = read(OUT / "fit_summary.json")
    require(read(journal)["status"] == "saved" and digest(model_path) == fit["model_sha256"] and
            fit["training"] == pre["training"] and fit["parameters"] == pre["parameters"], "Saved fit contract differs")
    model = lgb.Booster(model_str=model_path.read_text(encoding="utf-8"))
    validate_model(model, pre["training"], model_path.read_text(encoding="utf-8"))
    sites, price = final_inputs()
    spec = baseline_contract()
    fa, fb, excluded = load_forecasts("test", spec)
    (OUT / "months").mkdir(exist_ok=True)
    pieces, info = [], []
    for axis in monthly(FINAL.axis()):
        name = f"{axis[0]:%Y-%m}"
        path = OUT / "months" / f"{name}.parquet"
        meta_path = path.with_suffix(".json")
        if path.exists():
            m = read(meta_path)
            require(m["sha256"] == digest(path) and m["model_sha256"] == fit["model_sha256"], "Partial prediction drift")
            flat = pd.read_parquet(path)
            validate_site_grid(flat, sites, axis)
        else:
            d, x = frame(axis, sites, fa, fb, spec, pre["training"])
            raw = model.predict(x)
            values = prediction_values(d, raw)
            require((values[d.elev.to_numpy() <= 0] == 0).all(), "Night output differs")
            positions = np.r_[np.arange(253), np.arange(len(x)-253, len(x))]
            probe = x.iloc[positions]
            before = raw[positions]
            np.testing.assert_allclose(before, model.predict(probe.iloc[::-1])[::-1], rtol=1e-12, atol=0)
            batched = np.concatenate([model.predict(probe.iloc[z]) for z in np.array_split(np.arange(len(probe)), 3)])
            np.testing.assert_allclose(before, batched, rtol=1e-12, atol=0)
            flat = d[KEY].assign(kwh=values)
            validate_site_grid(flat, sites, axis)
            flat.to_parquet(path, index=False, compression="zstd")
            m = dict(month=name, rows=len(flat), feature_nan_rows=int(x.isna().any(axis=1).sum()),
                night_rows=int(d.elev.le(0).sum()), sha256=digest(path), model_sha256=fit["model_sha256"],
                batch_order_probe_rows=len(probe))
            write_json(meta_path, m)
            del d, x, raw, values, probe
        pieces.append(flat); info.append(m)
        log(f"P011 final inference: {name}")
    flat = pd.concat(pieces, ignore_index=True)
    validate_site_grid(flat, sites, FINAL.axis())
    flat.to_parquet(OUT / "site_predictions.parquet", index=False, compression="zstd")
    composition = sites[["site_id"]].assign(group=sites.current_group.map({"A": "G1", "B": "G2"}))
    composition.to_csv(OUT / "composition.csv", index=False, encoding="utf-8-sig")
    forecast = flat.assign(group=flat.site_id.map(composition.set_index("site_id").group)).groupby(["group", "ts"], as_index=False).kwh.sum()
    forecast = forecast.rename(columns={"kwh": "forecast_kwh"}).sort_values(["group", "ts"]).reset_index(drop=True)
    forecast.ts = forecast.ts.map(lambda ts: ts.isoformat(sep=" "))
    forecast.to_csv(OUT / "forecast.csv", index=False, encoding="utf-8-sig")
    saved_flat = pd.read_parquet(OUT / "site_predictions.parquet")
    pd.testing.assert_frame_equal(saved_flat, flat)
    check = validate_outputs(pd.read_csv(OUT / "composition.csv", encoding="utf-8-sig"),
        pd.read_csv(OUT / "forecast.csv", encoding="utf-8-sig", float_precision="round_trip"), saved_flat, sites, price)
    require(all((OUT / name).read_bytes().startswith(b"\xef\xbb\xbf") for name in ("composition.csv", "forecast.csv")), "CSV BOM missing")
    intact(pre)
    write_json(OUT / "summary.json", dict(status="passed", started_at=started, finished_at=now(), runtime=runtime(),
        chosen_method="P006_fixed_category", model_fit_count=1, fit_attempts=1, precheck_sha256=digest(OUT / "precheck.json"),
        fit=fit, window=vars(FINAL), output_validation=check, months=info, test_nonhourly=excluded,
        output_sha256={name: digest(OUT / name) for name in ("model.txt", "fit_summary.json", "site_predictions.parquet", "composition.csv", "forecast.csv")},
        block_summary_sha256=digest(BLOCK / "summary.json"), performance_scores_computed=False,
        V2_executed=False, official_submission_executed=False, preserved_B3_unchanged=True))
    log("P011 passed: one final fit, local outputs validated; no score or submission")


def verify():
    pre, summary = read(OUT / "precheck.json"), read(OUT / "summary.json")
    intact(pre)
    require(summary["status"] == "passed" and summary["precheck_sha256"] == digest(OUT / "precheck.json"), "Final summary drift")
    require(summary["model_fit_count"] == 1 and read(OUT / "fit_journal.json")["status"] == "saved", "Fit journal differs")
    for name, h in summary["output_sha256"].items():
        require(digest(OUT / name) == h, f"Final output hash drift: {name}")
    fit = read(OUT / "fit_summary.json")
    require(fit == summary["fit"] and fit["training"] == pre["training"], "Fit summary differs")
    model = lgb.Booster(model_str=(OUT / "model.txt").read_text(encoding="utf-8"))
    validate_model(model, pre["training"], (OUT / "model.txt").read_text(encoding="utf-8"))
    sites, price = final_inputs()
    flat = pd.read_parquet(OUT / "site_predictions.parquet")
    check = validate_outputs(pd.read_csv(OUT / "composition.csv", encoding="utf-8-sig"),
        pd.read_csv(OUT / "forecast.csv", encoding="utf-8-sig", float_precision="round_trip"), flat, sites, price)
    require(check == summary["output_validation"], "Final output validation differs")
    spec = baseline_contract()
    fa, fb, _ = load_forecasts("test", spec)
    probe_rows = 0
    for axis, item in zip(monthly(FINAL.axis()), summary["months"]):
        path = OUT / "months" / f"{axis[0]:%Y-%m}.parquet"
        require(digest(path) == item["sha256"] and read(path.with_suffix(".json")) == item, "Monthly manifest drift")
        month = pd.read_parquet(path)
        validate_site_grid(month, sites, axis)
        pd.testing.assert_frame_equal(month.reset_index(drop=True), flat.loc[flat.ts.isin(axis)].reset_index(drop=True))
        probe_axis = axis[np.unique([0, min(11, len(axis)-1), len(axis)-1])]
        d, x = frame(probe_axis, sites, fa, fb, spec, pre["training"])
        fresh = prediction_values(d, model.predict(x))
        saved = month.set_index(KEY).kwh.reindex(pd.MultiIndex.from_frame(d[KEY])).to_numpy()
        np.testing.assert_allclose(fresh, saved, rtol=1e-12, atol=0)
        # Verify every saved night cell without full model re-inference.
        meta = month.merge(sites[["site_id", "lat", "lon"]], on="site_id", how="left", validate="many_to_one")
        elev = spec["solar"](times(meta.ts), meta.lat.to_numpy(), meta.lon.to_numpy())[0]
        require((meta.kwh.to_numpy()[elev <= 0] == 0).all(), "Saved night output differs")
        probe_rows += len(x)
        log(f"P011 saved verification: {axis[0]:%Y-%m}")
    intact(pre)
    write_json(OUT / "verification.json", dict(status="passed", at=now(), new_fits=0, final_fit_attempts=1,
        fresh_probe_rows=probe_rows, full_output_validation_reproduced=True, all_saved_night_cells_zero=True,
        summary_sha256=digest(OUT / "summary.json"), preserved_B3_unchanged=True, performance_scores_computed=False))
    log("P011 separate saved verification passed")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    require(sys.version_info[:2] == (3, 13) and not sysconfig.get_config_var("Py_GIL_DISABLED"), "Regular CPython 3.13 required")
    {"--precheck": precheck, "--run": execute, "--resume-predictions": lambda: execute(True), "--verify": verify}[sys.argv[1]]()
