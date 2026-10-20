"""Pre-registered B2 V1 development plus T0 historical replay; never use V2 scores."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import platform
import sys
import sysconfig
import time
import unittest

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow

from b0_v1_baseline import feature_frame, model_parameters, prediction_values, target_values
from s0_validation_check import (ROOT, SOURCE, INPUTS, baseline_contract, digest, load_forecasts,
                                 log, monthly, now, write_json)
from validation_core import (KST, KEY, WINDOWS, Window, clock, group_totals, grid, local_score,
                             metadata, require, times, truth_matrix, unique, weights)
from v1_residual_diagnostics import frozen_artifacts

OUT = ROOT / "data/validation/2026-10-20_B2_joint"
REPORT = ROOT / "reports/2026-10-20_03_B2_공동입력_검증설계와_비교.md"
T0 = Window("T0", "2023-01-01", "2024-01-01", "2022-01-01", "2022-12-31")
B0 = ROOT / "data/validation/2026-10-19_B0_V1"


def joint_features(d10, x10, d17, x17, spec):
    require(pd.MultiIndex.from_frame(d10[KEY]).equals(pd.MultiIndex.from_frame(d17[KEY])), "Round key/order mismatch")
    weather, common = spec["FEATS"][:15], spec["FEATS"][15:]
    pd.testing.assert_frame_equal(x10[common], x17[common])
    if "energy" in d10:
        np.testing.assert_array_equal(target_values(d10), target_values(d17))
    result = pd.concat([x10[weather].add_suffix("_r10"), x17[weather].add_suffix("_r17"), x10[common]], axis=1)
    require(result.shape == (len(d10), 39) and not result.columns.duplicated().any(), "Joint feature contract mismatch")
    require(not np.isinf(result.to_numpy()).any(), "Infinite joint features")
    return result


def frames(base, fa, fb, spec):
    d10, x10 = feature_frame(base, fa, fb, 10, spec)
    d17, x17 = feature_frame(base, fa, fb, 17, spec)
    return d10, x10, d17, x17


def scores(truth, pred, groups, price, ids, cohort_name):
    rows = []
    start = truth.index-pd.Timedelta(hours=1)
    masks = [("all", np.ones(len(start), dtype=bool))]
    masks += [(f"Q{q}", start.quarter == q) for q in range(1, 5)]
    masks += [(f"M{m:02}", start.month == m) for m in range(1, 13)]
    masks += [(f"k={k}", weights(truth.index) == k) for k in (1, 3)]
    for name, members in ((cohort_name, ids), ("common_253", truth.columns.tolist())):
        a, complete = group_totals(truth[members], groups)
        y, _ = group_totals(pred[members], groups)
        for period, mask in masks:
            axis = truth.index[complete.to_numpy() & mask]
            if not len(axis):
                continue
            r = local_score(truth[members], pred[members], groups, price, axis)
            q = price.set_index("ts").loc[axis].smp.to_numpy()*weights(axis)
            direct = float(np.sum(np.abs(y.loc[axis].to_numpy()-a.loc[axis].to_numpy())*q[:, None]))
            require(np.isclose(direct, r["W_local"], rtol=1e-12, atol=1e-5), "Independent W mismatch")
            rows.append(dict(sample=name, period=period, first_ts=str(axis.min()), last_ts=str(axis.max()), **r))
        part = [r for r in rows if r["sample"] == name]
        if part:
            total = next(r for r in part if r["period"] == "all")
            for prefix in ("M", "Q", "k="):
                split = [r for r in part if r["period"].startswith(prefix)]
                require(sum(r["times"] for r in split) == total["times"], "Score time partition mismatch")
                for field in ("W_local", "R_local", "S_local", "price_k_sum"):
                    require(np.isclose(sum(r[field] for r in split), total[field], rtol=1e-12), "Score partition mismatch")
    return rows


def compare(old, new, cohort_name):
    a = {(r["sample"], r["period"]): r for r in old}
    b = {(r["sample"], r["period"]): r for r in new}
    require(set(a) == set(b), "Different score scope")
    rows = []
    for key, r0 in a.items():
        r2 = b[key]
        for field in ("times", "first_ts", "last_ts", "price_k_sum", "R_local"):
            require(r0[field] == r2[field], f"Different score sample: {field}")
        w0, w2 = r0["W_local"], r2["W_local"]
        rows.append(dict(sample=key[0], period=key[1], times=r0["times"], B0_W=w0, B2_W=w2,
                         delta_W=w2-w0, change_pct=(w2/w0-1)*100 if w0 else None,
                         B0_mae=r0["weighted_group_mae_kwh"], B2_mae=r2["weighted_group_mae_kwh"]))
    main = next(r for r in rows if r["sample"] == cohort_name and r["period"] == "all")
    quarters = [r for r in rows if r["sample"] == cohort_name and r["period"].startswith("Q")]
    require(len(quarters) == 4, "Need four quarters")
    improved = main["B2_W"] < main["B0_W"] and not np.isclose(main["B2_W"], main["B0_W"], rtol=1e-12, atol=0)
    unstable = [r["period"] for r in quarters if
                (r["B0_W"] == 0 and r["B2_W"] > 0) or
                (r["B0_W"] > 0 and r["B2_W"] > r["B0_W"]*1.05 and
                 not np.isclose(r["B2_W"], r["B0_W"]*1.05, rtol=1e-12, atol=0))]
    return dict(annual_improved=bool(improved), unstable_quarters=unstable,
                passes=bool(improved and not unstable), rows=rows)


def runtime():
    return dict(executable=sys.executable, python=platform.python_version(), lightgbm=lgb.__version__,
                numpy=np.__version__, pandas=pd.__version__, pyarrow=pyarrow.__version__, free_threaded=False)


def code_paths():
    return [Path(__file__), ROOT/"scripts/b0_v1_baseline.py", ROOT/"scripts/s0_validation_check.py",
            ROOT/"scripts/validation_core.py", ROOT/"scripts/v1_residual_diagnostics.py",
            ROOT/"tests/test_b2_validation.py"]


def run_precheck():
    require(not (OUT/"precheck.json").exists(), "Precheck already frozen")
    OUT.mkdir(parents=True, exist_ok=True)
    spec = baseline_contract()
    source = {f: digest(SOURCE/f) for f in INPUTS}
    b0 = json.loads((B0/"summary.json").read_text(encoding="utf-8"))
    require(source == b0["source_sha256"] and runtime() == b0["runtime"] and model_parameters() == b0["model_parameters"],
            "B0 source/runtime/settings drift")
    sites = pd.read_csv(SOURCE/"sites.csv", encoding="utf-8-sig")
    metadata(sites)
    gen = pd.read_parquet(SOURCE/"train/generation.parquet")
    gen["ts"] = times(gen.ts)
    unique(gen, KEY)
    train = T0.training(gen)
    truth = truth_matrix(gen, b0["fixed_148_sites"], T0.axis())
    ids = truth.columns[truth.notna().all(axis=0)].tolist()
    cohort = sites.set_index("site_id").loc[ids]
    require(set(cohort.current_group) == {"A", "B"}, "T0 cohort lacks both reference groups")
    d = train.merge(sites, on="site_id", how="left", validate="many_to_one")
    elev = spec["solar"](pd.DatetimeIndex(d.ts), d.lat.to_numpy(), d.lon.to_numpy())[0]
    nday = int((elev > 0).sum())
    require(nday >= 10000, "Insufficient T0 daylight rows")
    whole = truth_matrix(gen, sites.site_id, T0.axis())
    price = pd.read_csv(SOURCE/"train/smp.csv", encoding="utf-8-sig")
    price["ts"] = times(price.ts)
    unique(price, ["ts"])
    p = price.set_index("ts").reindex(T0.axis()).smp
    require(np.isfinite(p).all() and p.ge(0).all(), "T0 price missing")
    codes = {str(p.relative_to(ROOT)): digest(p) for p in code_paths()}
    protected = frozen_artifacts()
    value = dict(status="passed", at=now(), runtime=runtime(), source_sha256=source,
        protected_sha256=protected, prereg_report_sha256=digest(REPORT), code_sha256=codes,
        T0=dict(window=vars(T0), train_rows=len(train), train_sites=int(train.site_id.nunique()),
                daylight_rows=nday, cohort_ids=ids, cohort_sites=len(ids),
                cohort_train_sites=int(cohort.index.isin(train.site_id.unique()).sum()),
                group_sites=cohort.groupby("current_group").size().to_dict(),
                group_capacity_kw={k: float(v) for k,v in cohort.groupby("current_group").capacity_kw.sum().items()},
                common253_times=int(whole.notna().all(axis=1).sum()), prediction_times=len(T0.axis()),
                prediction_rows=253*len(T0.axis()), observed_truth_rows=int(whole.notna().to_numpy().sum()),
                price_k_sum=float((p*weights(T0.axis())).sum()), first_train_ts=str(train.ts.min()),last_train_ts=str(train.ts.max())),
        V1_cohort_ids=b0["fixed_148_sites"], model_fit_budget=4, model_fit_count=0, scores_computed=False)
    (OUT/"executed_code").mkdir(exist_ok=True)
    for path in code_paths():
        (OUT/"executed_code"/path.name).write_bytes(path.read_bytes())
    (OUT/"preregistered_report.md").write_bytes(REPORT.read_bytes())
    require(source == {f:digest(SOURCE/f) for f in INPUTS} and protected == frozen_artifacts(), "Inputs changed")
    write_json(OUT/"precheck.json", value)
    log(f"Precheck passed without fit/score; T0 train={len(train):,}, daylight={nday:,}, cohort={len(ids)}, groups={value['T0']['group_sites']}")


def fit_predict(window, method, gen, sites, fa, fb, spec, params, target, output_root=OUT):
    folder = output_root/window.name/method
    folder.mkdir(parents=True, exist_ok=True)
    require(not (folder/"fit_summary.json").exists(), "Completed fits exist; no refit")
    train = window.training(gen)
    base = train.merge(sites, on="site_id", how="left", validate="many_to_one", sort=False)
    preds, fits = [], []
    for mode in ((10,17) if method == "B0" else ("joint",)):
        if mode == "joint":
            d, x10, d17, x17 = frames(base, fa, fb, spec)
            x = joint_features(d, x10, d17, x17, spec)
            del x10, d17, x17
        else:
            d, x = feature_frame(base, fa, fb, mode, spec)
        day = d.elev.gt(0)
        x, y = x.loc[day], target_values(d.loc[day])
        log(f"{window.name} {method} {mode}: fit {x.shape}")
        t0 = time.perf_counter()
        model = lgb.LGBMRegressor(**params).fit(x, y)
        require(model.booster_.num_trees() == 600, "Unexpected tree count")
        # Native LightGBM file IO cannot write this Windows non-ASCII path.
        model_file = folder/f"model_{mode}.txt"
        model_file.write_text(model.booster_.model_to_string(), encoding="utf-8")
        info = dict(mode=mode, train_rows=len(base), daylight_rows=len(x), features=x.columns.tolist(),
                    fit_seconds=time.perf_counter()-t0, model_sha256=digest(model_file),
                    train_ts_min=str(base.ts.min()), train_ts_max=str(base.ts.max()),
                    any_feature_nan_rows=int(x.isna().any(axis=1).sum()))
        del d,x,y
        gc.collect()
        pieces = []
        for axis in monthly(window.axis()):
            b = grid(sites, axis)
            if mode == "joint":
                d,x10,d17,x17 = frames(b,fa,fb,spec)
                x = joint_features(d,x10,d17,x17,spec)
                del x10,d17,x17
            else:
                d,x = feature_frame(b,fa,fb,mode,spec)
            d["kwh"] = prediction_values(d,model.predict(x))
            pieces.append(d[KEY+["kwh"]].copy())
            log(f"{window.name} {method} {mode}: predicted {axis[0]:%Y-%m}, rows={len(d):,}")
        flat = pd.concat(pieces,ignore_index=True)
        unique(flat,KEY)
        require(len(flat) == len(window.axis())*253, "Prediction rows lost")
        flat.to_parquet(folder/f"predictions_{mode}.parquet",index=False,compression="zstd")
        matrix = flat.pivot(index="ts",columns="site_id",values="kwh").reindex(index=target.index,columns=target.columns)
        require(np.isfinite(matrix.to_numpy()).all() and matrix.ge(0).all().all(), "Incomplete predictions")
        preds.append(matrix)
        fits.append(info)
        del model,pieces,flat,d,x
        gc.collect()
    prediction = sum(preds)/len(preds)
    prediction.rename_axis(index="ts",columns="site_id").stack().rename("kwh").reset_index().to_parquet(
        folder/"predictions_mean.parquet",index=False,compression="zstd")
    write_json(folder/"fit_summary.json",fits)
    return prediction,fits


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    require(sys.version_info[:2] == (3,13) and not sysconfig.get_config_var("Py_GIL_DISABLED"), "Regular CPython 3.13 required")
    suite = unittest.defaultTestLoader.discover(str(ROOT/"tests"),pattern="test_b2_validation.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    require(result.wasSuccessful(), "B2 contracts failed")
    if "--precheck" in sys.argv:
        run_precheck()
        return
    require(not (OUT/"summary.json").exists(), "Completed experiment exists")
    pre = json.loads((OUT/"precheck.json").read_text(encoding="utf-8"))
    require(pre["status"] == "passed" and pre["runtime"] == runtime(), "Precheck mismatch")
    require(pre["prereg_report_sha256"] == digest(REPORT), "Preregistered report changed before execution")
    for rel,h in pre["code_sha256"].items():
        require(digest(ROOT/rel) == h, "Code changed after precheck")
    require(pre["source_sha256"] == {f:digest(SOURCE/f) for f in INPUTS}, "Source changed")
    require(pre["protected_sha256"] == frozen_artifacts(), "Existing baseline changed")
    started = now()
    sites = pd.read_csv(SOURCE/"sites.csv",encoding="utf-8-sig")
    metadata(sites)
    groups = sites.set_index("site_id").current_group.map({"A":"G1","B":"G2"})
    gen = pd.read_parquet(SOURCE/"train/generation.parquet")
    gen["ts"] = times(gen.ts)
    unique(gen,KEY)
    price = pd.read_csv(SOURCE/"train/smp.csv",encoding="utf-8-sig")
    price["ts"] = times(price.ts)
    spec,params = baseline_contract(),model_parameters()
    fa,fb,excluded = load_forecasts("train",spec)
    outputs = []
    # Always run both periods with fixed settings, regardless of first-period score.
    for window in (WINDOWS[0],T0):
        truth = truth_matrix(gen,sites.site_id,window.axis())
        ids = pre["V1_cohort_ids"] if window.name == "V1" else pre["T0"]["cohort_ids"]
        name = "fixed_148" if window.name == "V1" else "complete_T0_cohort"
        require(truth[ids].notna().all().all(), "Frozen cohort truth missing")
        if window.name == "V1":
            old = json.loads((B0/"summary.json").read_text(encoding="utf-8"))
            flat = pd.read_parquet(B0/"predictions_mean.parquet")
            b0pred = flat.pivot(index="ts",columns="site_id",values="kwh").reindex(index=truth.index,columns=truth.columns)
            base_fits=[]
        else:
            b0pred,base_fits = fit_predict(window,"B0",gen,sites,fa,fb,spec,params,truth)
        baseline_scores = scores(truth,b0pred,groups,price,ids,name)
        if window.name == "V1":
            ref = {(r["sample"],r["period"]):r for r in old["group_scores"]}
            for r in baseline_scores:
                for key in ("W_local","R_local","S_local","price_k_sum","times"):
                    require(np.isclose(r[key],ref[(r["sample"],r["period"])][key],rtol=1e-12,atol=1e-5),"B0 V1 score drift")
        joint,joint_fits = fit_predict(window,"B2",gen,sites,fa,fb,spec,params,truth)
        new_scores=scores(truth,joint,groups,price,ids,name)
        decision=compare(baseline_scores,new_scores,name)
        folder=OUT/window.name
        pd.DataFrame(decision["rows"]).to_csv(folder/"comparison.csv",index=False,encoding="utf-8-sig")
        outputs.append(dict(window=vars(window),cohort_name=name,cohort_sites=len(ids),
                            baseline_scores=baseline_scores,joint_scores=new_scores,
                            fits=base_fits+joint_fits,comparison=decision))
        log(f"{window.name}: fixed comparison completed")
    fit_count=sum(len(o["fits"]) for o in outputs)
    require(fit_count == 4,"Fit budget mismatch")
    require(pre["protected_sha256"] == frozen_artifacts(),"Baseline/EDA changed")
    require(pre["source_sha256"] == {f:digest(SOURCE/f) for f in INPUTS},"Source changed")
    require(pre["code_sha256"] == {str(p.relative_to(ROOT)):digest(p) for p in code_paths()},"Code changed during execution")
    passes=all(o["comparison"]["passes"] for o in outputs)
    write_json(OUT/"summary.json",dict(status="passed",experiment="B2_joint_V1_T0",started_at=started,finished_at=now(),
        runtime=runtime(),precheck_sha256=digest(OUT/"precheck.json"),synthetic_tests=result.testsRun,
        model_parameters=params,model_fit_count=fit_count,V2_executed=False,final_evaluation_executed=False,
        selection="B2_promising_candidate" if passes else "retain_B0",periods=outputs,
        protected_unchanged=True,source_unchanged=True,nonhourly=excluded))
    log(f"B2 fixed comparison passed; fits=4; selection={'B2 candidate' if passes else 'retain B0'}; no V2")


if __name__ == "__main__":
    main()
