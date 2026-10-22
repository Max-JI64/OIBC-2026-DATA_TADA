"""P-006: fixed B3 plus one nominal site feature; four candidate fits only."""
from __future__ import annotations

import gc
import hashlib
from pathlib import Path
import re
import sys
import sysconfig
import time
import traceback

import lightgbm as lgb
import numpy as np
import pandas as pd

from b0_v1_baseline import target_values, prediction_values
from b2_joint_compare import frames, joint_features, runtime, scores
from b3_l1_compare import parameters
from forecast_residual_eda import cohort_ids, pred_path, read
from h1_cloud_interaction import WINDOWS, inputs, development_forecasts, reference, load_prediction, gate as old_gate
from s0_validation_check import ROOT, SOURCE, baseline_contract, digest, log, monthly, now, write_json
from validation_core import KEY, clock, grid, require, truth_matrix, weights

OUT = ROOT / "data/validation/2026-10-22_P006_site_category"
REPORT = ROOT / "reports/2026-10-22_03_P006_발전소ID_범주입력_비교.md"
PLAN = ROOT / "reports/성능개선_진행계획.md"
HISTORY = ROOT / "data/validation/2026-10-21_H1_cloud_interaction/precheck.json"
NEW = "site_id_category"
CAT_PARAMS = dict(cat_smooth=10.0, cat_l2=10.0, min_data_per_group=100,
                  max_cat_threshold=32, max_cat_to_onehot=4)


def candidate_parameters():
    return dict(**parameters(), **CAT_PARAMS)


def categories_from_training(ids):
    require(ids.notna().all() and len(ids) > 0, "Empty/missing training IDs")
    return sorted(ids.unique().tolist())


def categorical_features(x, ids, categories):
    require(x.shape[1] == 39 and NEW not in x, "Expected original 39 features")
    require(x.index.equals(ids.index), "Site/feature row alignment differs")
    require(x.dtypes.eq("float32").all() and not np.isinf(x.to_numpy()).any(), "Original schema differs")
    require(ids.notna().all(), "Missing site key")
    require(len(categories) > 0 and categories == sorted(set(categories)), "Invalid frozen dictionary")
    result = x.copy()
    result[NEW] = pd.Categorical(ids.where(ids.isin(categories)), categories=categories, ordered=False)
    pd.testing.assert_frame_equal(result.iloc[:, :39], x)
    require(result[NEW].isna().equals(~ids.isin(categories)), "Unknown-ID mask differs")
    require(isinstance(result[NEW].dtype, pd.CategoricalDtype) and not result[NEW].cat.ordered,
            "Nominal category lost")
    return result


def gate(before, after, annual):
    value = old_gate(before, after, "fixed_cohort", annual)
    for row in value["rows"]:
        row["P006_W"] = row.pop("H1_W")
    return value


def code_paths():
    return [Path(__file__), ROOT / "tests/test_p006_site_category.py"] + [ROOT / "scripts" / name for name in
        ("h1_cloud_interaction.py", "b0_v1_baseline.py", "b2_joint_compare.py", "b3_l1_compare.py",
         "forecast_residual_eda.py", "s0_validation_check.py", "validation_core.py")]


def protected():
    """Verify specific B3 inputs/artifacts against frozen history, not old live plans."""
    historical = read(HISTORY)
    expected = historical["protected_sha256"]
    paths = [SOURCE / p for p in ("sites.csv", "baseline.py", "train/generation.parquet",
             "train/forecast_wa.parquet", "train/forecast_wb.parquet", "train/smp.csv")]
    paths += [ROOT / "data/validation/2026-10-20_B2_joint/precheck.json",
              ROOT / "data/validation/2026-10-20_B3_B5_rolling/availability.json"]
    for w in WINDOWS:
        folder = pred_path(w.name, "B3").parent
        paths += [pred_path(w.name, "B3"), folder / "fit_summary.json", folder.parents[1] / "summary.json",
                  folder / ("model_joint.txt" if w.name in ("T0", "V1") else "model.txt")]
    current = {}
    for p in set(paths):
        rel = str(p.relative_to(ROOT))
        require(rel in expected and digest(p) == expected[rel], f"Historical B3 input/artifact drift: {rel}")
        current[rel] = expected[rel]
    for rel, h in historical["code_sha256"].items():
        if rel.startswith("scripts"):
            require(digest(ROOT / rel) == h, f"Historical feature/execution code drift: {rel}")
            current[rel] = h
    current[str(HISTORY.relative_to(ROOT))] = digest(HISTORY)
    return current


def training(w, gen, sites, fa, fb, spec, ref):
    t = w.training(gen)
    base = t.merge(sites, on="site_id", how="left", validate="many_to_one", sort=False)
    d, x10, d17, x17 = frames(base, fa, fb, spec)
    x = joint_features(d, x10, d17, x17, spec)
    day = d.elev.gt(0)
    orig = x.loc[day]
    c = dict(train_rows=len(t), daylight_rows=int(day.sum()), train_ts_min=str(t.ts.min()),
             train_ts_max=str(t.ts.max()), features=orig.columns.tolist(), dtypes=list(map(str, orig.dtypes)),
             nan_rows=int(orig.isna().any(axis=1).sum()))
    require(c == ref["contract"], "Training differs from frozen B3 contract")
    y = target_values(d.loc[day])
    target_hash = hashlib.sha256(y.to_numpy().tobytes()).hexdigest()
    prior = next(p for p in read(HISTORY)["periods"] if p["window"]["name"] == w.name)
    require(target_hash == prior["training"]["raw_target_sha256"], "Historical training target/order drift")
    ids = d.loc[day, "site_id"]
    categories = categories_from_training(ids)
    xx = categorical_features(orig, ids, categories)
    require(xx.index.equals(y.index) and xx[NEW].notna().all(), "Training alignment/categories differ")
    detail = []
    for site_id, z in d.loc[day, ["site_id", "ts"]].groupby("site_id", sort=True):
        detail.append(dict(site_id=site_id, daylight_rows=len(z), first_ts=str(z.ts.min()), last_ts=str(z.ts.max()),
                           months=int(z.ts.dt.strftime("%Y-%m").nunique())))
    audit = dict(**c, categories=categories, new_features=xx.columns.tolist(), new_dtypes=list(map(str, xx.dtypes)),
                 raw_target_sha256=target_hash, original_x_sha256=hashlib.sha256(orig.to_numpy().tobytes()).hexdigest(),
                 ordered_site_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                 per_site=detail, first_prediction_at=str(clock(w.axis()).prediction_at.min()))
    del t, base, d, x10, d17, x17, x, orig
    gc.collect()
    return xx, y, audit


def precheck():
    OUT.mkdir(parents=True, exist_ok=True)
    require(not (OUT / "precheck.json").exists(), "Precheck already frozen")
    hashes = protected()
    sites, gen, price = inputs()
    spec = baseline_contract()
    fa, fb, excluded = development_forecasts(spec)
    periods = []
    for w in WINDOWS:
        ref = reference(w)
        x, y, audit = training(w, gen, sites, fa, fb, spec, ref)
        ids = cohort_ids(w.name)
        truth = truth_matrix(gen, sites.site_id, w.axis())
        require(truth[ids].notna().all().all(), "Fixed cohort truth incomplete")
        p = price.set_index("ts").smp.reindex(w.axis())
        require(np.isfinite(p).all() and p.ge(0).all(), "Price coverage invalid")
        categories = audit["categories"]
        probe_ids = pd.Series(sites.site_id.tolist(), name="site_id")
        probe = pd.DataFrame(np.zeros((len(sites), 39), dtype="float32"), columns=audit["features"])
        encoded = categorical_features(probe, probe_ids, categories)
        require(encoded[NEW].isna().sum() == len(set(sites.site_id) - set(categories)), "Prediction dictionary mismatch")
        periods.append(dict(window=vars(w), cohort_ids=ids, cohort_sites=len(ids), training=audit, reference=ref,
                            known_train_sites=len(categories), unseen_grid_sites=len(set(sites.site_id)-set(categories)),
                            unseen_cohort_sites=len(set(ids)-set(categories)),
                            common253_times=int(truth.notna().all(axis=1).sum())))
        log(f"P006 precheck {w.name}: {len(x)} daylight rows, {len(categories)} known IDs; no fits")
        del x, y, truth
        gc.collect()
    require(protected() == hashes, "Protected files changed during precheck")
    (OUT / "executed_code").mkdir(exist_ok=True)
    for p in code_paths():
        (OUT / "executed_code" / p.name).write_bytes(p.read_bytes())
    (OUT / "preregistered_report.md").write_bytes(REPORT.read_bytes())
    (OUT / "performance_plan.md").write_bytes(PLAN.read_bytes())
    write_json(OUT / "precheck.json", dict(at=now(), runtime=runtime(), parameters=candidate_parameters(),
        periods=periods, protected_sha256=hashes, code_sha256={str(p.relative_to(ROOT)): digest(p) for p in code_paths()},
        report_sha256=digest(REPORT), plan_sha256=digest(PLAN), fit_budget=4, model_fit_count=0,
        reference_refit_count=0, excluded_nonhourly=excluded))
    log("P006 precheck passed; all four B3 references reusable")


def intact(pre, live=True):
    require(pre["runtime"] == runtime() and pre["parameters"] == candidate_parameters(), "Runtime/settings drift")
    require(pre["protected_sha256"] == protected(), "Protected files drift")
    for rel, h in pre["code_sha256"].items():
        require(digest(ROOT / rel) == digest(OUT / "executed_code" / Path(rel).name) == h, "Frozen code drift")
    require(digest(OUT / "preregistered_report.md") == pre["report_sha256"] and
            digest(OUT / "performance_plan.md") == pre["plan_sha256"], "Frozen scope drift")
    if live:
        require(digest(REPORT) == pre["report_sha256"] and digest(PLAN) == pre["plan_sha256"], "Live scope drift")


def validate_model(model, audit, text):
    require(model.num_trees() == 600 and model.feature_name() == audit["new_features"], "Saved tree/schema drift")
    require(model.dump_model()["objective"] == "regression_l1", "Saved objective differs")
    require(model.pandas_categorical == [audit["categories"]], "Saved categorical dictionary differs")
    for key, value in CAT_PARAMS.items():
        match = re.search(r"\[" + key + r": ([^\]]+)\]", text)
        require(match is not None and float(match[1]) == value, f"Effective categorical parameter differs: {key}")
    categorical = re.search(r"\[categorical_feature: ([^\]]+)\]", text)
    require(categorical is not None and categorical[1] == "39", "Categorical feature not declared at fit")


def prediction_frame(w, axis, sites, fa, fb, spec, audit):
    d, x10, d17, x17 = frames(grid(sites, axis), fa, fb, spec)
    x = categorical_features(joint_features(d, x10, d17, x17, spec), d.site_id, audit["categories"])
    require(x.columns.tolist() == audit["new_features"], "Prediction schema differs")
    return d, x


def fit_predict(w, av, gen, sites, fa, fb, spec, truth):
    folder = OUT / w.name
    folder.mkdir(exist_ok=True)
    infofile, modelpath, journal = [folder / n for n in ("fit_summary.json", "model.txt", "fit_journal.json")]
    if not infofile.exists():
        require(not journal.exists() and not modelpath.exists(), "Partial fit exists; no automatic refit")
        x, y, audit = training(w, gen, sites, fa, fb, spec, av["reference"])
        require(audit == av["training"], "Training frozen audit differs")
        write_json(journal, dict(status="started", at=now(), fit_attempts=1))
        started = time.perf_counter()
        log(f"{w.name}: P006 fit starts {x.shape}, one attempt")
        try:
            estimator = lgb.LGBMRegressor(**candidate_parameters()).fit(x, y, categorical_feature=[NEW])
            fit_seconds = time.perf_counter()-started
            text = estimator.booster_.model_to_string()
            modelpath.write_text(text, encoding="utf-8")
            reloaded = lgb.Booster(model_str=modelpath.read_text(encoding="utf-8"))
            validate_model(reloaded, audit, text)
            # Probe all 253 sites at several development hours, including unseen IDs.
            axis = w.axis()[[0, 11, 23, len(w.axis())-1]]
            d, probe = prediction_frame(w, axis, sites, fa, fb, spec, audit)
            before = estimator.booster_.predict(probe)
            after = reloaded.predict(probe)
            np.testing.assert_allclose(before, after, rtol=1e-12, atol=0)
            shuffled = reloaded.predict(probe.iloc[::-1])[::-1]
            np.testing.assert_allclose(after, shuffled, rtol=1e-12, atol=0)
            batched = np.concatenate([reloaded.predict(probe.iloc[index])
                                      for index in np.array_split(np.arange(len(probe)), 3)])
            np.testing.assert_allclose(after, batched, rtol=1e-12, atol=0)
            info = dict(training=audit, parameters=candidate_parameters(), model_sha256=digest(modelpath),
                        fit_seconds=fit_seconds, fit_attempts=1, model_fit_count=1,
                        reload_probe_rows=len(probe), reload_batch_order_equal=True,
                        id_split_count=int(reloaded.feature_importance("split")[-1]),
                        id_gain=float(reloaded.feature_importance("gain")[-1]))
            write_json(infofile, info)
            write_json(journal, dict(status="saved", at=now(), fit_attempts=1))
            del estimator, reloaded, x, y, probe, d
            gc.collect()
        except BaseException:
            write_json(journal, dict(status="failed", at=now(), fit_attempts=1, error=traceback.format_exc()))
            raise
    info = read(infofile)
    require(info["training"] == av["training"] and info["parameters"] == candidate_parameters() and
            digest(modelpath) == info["model_sha256"], "Completed candidate fit differs")
    text = modelpath.read_text(encoding="utf-8")
    model = lgb.Booster(model_str=text)
    validate_model(model, av["training"], text)
    path = folder / "predictions.parquet"
    if not path.exists():
        parts = []
        for axis in monthly(w.axis()):
            d, x = prediction_frame(w, axis, sites, fa, fb, spec, av["training"])
            parts.append(d[KEY].assign(kwh=prediction_values(d, model.predict(x))))
            log(f"{w.name}: inferred {axis[0]:%Y-%m}")
            del d, x
            gc.collect()
        pd.concat(parts, ignore_index=True).to_parquet(path, index=False, compression="zstd")
        write_json(folder / "prediction_manifest.json", dict(sha256=digest(path), model_sha256=info["model_sha256"]))
        del parts
    require(read(folder / "prediction_manifest.json") == dict(sha256=digest(path), model_sha256=info["model_sha256"]),
            "Candidate prediction hash drift")
    pred = load_prediction(path, truth)
    del model
    gc.collect()
    return pred, info


def diagnostics(truth, pred, sites, price, ids, categories):
    groups = sites.set_index("site_id").current_group.map({"A": "G1", "B": "G2"})
    fixed, known = set(cohort_ids("V1")), set(categories)
    subsets = [("complete", ids), ("fixed_148", [i for i in ids if i in fixed]),
               ("added_complete", [i for i in ids if i not in fixed]),
               ("train_seen", [i for i in ids if i in known]), ("train_unseen", [i for i in ids if i not in known])]
    q = price.set_index("ts").smp.reindex(truth.index).to_numpy() * weights(truth.index)
    rows, coverage = [], []
    for cohort, members in subsets:
        if not members:
            coverage.append(dict(cohort=cohort, sites=0, W_available=False, reason="empty"))
            continue
        require(truth[members].notna().all().all(), "Diagnostic fixed cohort incomplete")
        both = set(groups[members]) == {"G1", "G2"}
        coverage.append(dict(cohort=cohort, sites=len(members), W_available=both,
                             individual_mae_kwh=float((pred[members]-truth[members]).abs().to_numpy().mean())))
        if not both:
            continue
        for group in ("G1", "G2"):
            gi = [i for i in members if groups[i] == group]
            e = (pred[gi].sum(axis=1)-truth[gi].sum(axis=1)).to_numpy()
            for k in ("all", 1, 3):
                mask = np.ones(len(q), bool) if k == "all" else weights(truth.index) == k
                qq, ee = q[mask], e[mask]
                rows.append(dict(cohort=cohort, group=group, k=str(k), sites=len(gi), times=int(mask.sum()),
                                 W=float(np.sum(qq*np.abs(ee))), over_W=float(np.sum(qq*np.maximum(ee, 0))),
                                 under_W=float(np.sum(qq*np.maximum(-ee, 0)))))
            part = [r for r in rows if r["cohort"] == cohort and r["group"] == group]
            whole = part[0]
            for field in ("W", "over_W", "under_W"):
                require(np.isclose(sum(r[field] for r in part[1:]), whole[field], rtol=1e-12), "Diagnostic k sum differs")
            require(np.isclose(whole["W"], whole["over_W"]+whole["under_W"], rtol=1e-12), "Direction sum differs")
    # Unseen sites may have sparse truth outside the fixed primary cohort: individual errors only.
    sparse = []
    for label, members in (("seen", [i for i in truth if i in known]), ("unseen", [i for i in truth if i not in known])):
        valid = truth[members].notna()
        count = int(valid.to_numpy().sum())
        abs_error = (pred[members]-truth[members]).abs()
        sparse.append(dict(category=label, sites=len(members), observed_site_hours=count,
                           individual_mae_kwh=float(abs_error.sum().sum()/count) if count else None))
    return dict(group_rows=rows, cohort_coverage=coverage, observed_id_errors=sparse)


def reference_scores(truth, old, groups, price, av):
    result = scores(truth, old, groups, price, av["cohort_ids"], "fixed_cohort")
    historical = {(r["period"], r["sample"] == "common_253"): r for r in av["reference"]["scores"]}
    for row in result:
        prior = historical[(row["period"], row["sample"] == "common_253")]
        for key in ("W_local", "R_local", "S_local", "price_k_sum", "times"):
            require(np.isclose(row[key], prior[key], rtol=1e-12, atol=1e-5), "B3 score reproduction failed")
    return result


def execute():
    require(not (OUT / "summary.json").exists(), "Experiment complete; use --verify")
    pre = read(OUT / "precheck.json")
    intact(pre)
    started = now()
    sites, gen, price = inputs()
    spec = baseline_contract()
    fa, fb, _ = development_forecasts(spec)
    groups = sites.set_index("site_id").current_group.map({"A": "G1", "B": "G2"})
    output = []
    for w, av in zip(WINDOWS, pre["periods"]):
        require(vars(w) == av["window"], "Window drift")
        truth = truth_matrix(gen, sites.site_id, w.axis())
        old = load_prediction(pred_path(w.name, "B3"), truth)
        before = reference_scores(truth, old, groups, price, av)
        new, fit = fit_predict(w, av, gen, sites, fa, fb, spec, truth)
        after = scores(truth, new, groups, price, av["cohort_ids"], "fixed_cohort")
        comparison = gate(before, after, w.name in ("T0", "V1"))
        diag = {name: diagnostics(truth, pred, sites, price, av["cohort_ids"], av["training"]["categories"])
                for name, pred in (("B3", old), ("P006", new))}
        for name, ss in (("B3", before), ("P006", after)):
            total = next(r["W_local"] for r in ss if r["sample"] == "fixed_cohort" and r["period"] == "all")
            group_sum = sum(r["W"] for r in diag[name]["group_rows"] if r["cohort"] == "complete" and r["k"] == "all")
            require(np.isclose(total, group_sum, rtol=1e-12), "Group W sum differs")
        value = dict(window=vars(w), fit=fit, B3_scores=before, P006_scores=after, comparison=comparison, diagnostics=diag)
        write_json(OUT / w.name / "result.json", value)
        output.append(value)
        log(f"{w.name}: complete; passes={comparison['passes']}")
        del truth, old, new
        gc.collect()
    intact(pre)
    require(sum(p["fit"]["fit_attempts"] for p in output) == 4, "Fit budget differs")
    write_json(OUT / "summary.json", dict(status="passed", experiment="P006", started_at=started, finished_at=now(),
        model_fit_count=4, fit_attempts=4, reference_refit_count=0, runtime=runtime(),
        selection="P006" if all(p["comparison"]["passes"] for p in output) else "B3", periods=output,
        precheck_sha256=digest(OUT / "precheck.json"), V2_executed=False, evaluation_executed=False))
    log("P006 complete: four fits; run saved verification next")


def verify():
    pre, summary = read(OUT / "precheck.json"), read(OUT / "summary.json")
    intact(pre, live=False)
    require(summary["precheck_sha256"] == digest(OUT / "precheck.json") and summary["model_fit_count"] == 4,
            "Completion differs")
    sites, gen, price = inputs()
    spec = baseline_contract()
    fa, fb, _ = development_forecasts(spec)
    groups = sites.set_index("site_id").current_group.map({"A": "G1", "B": "G2"})
    for w, av, value in zip(WINDOWS, pre["periods"], summary["periods"]):
        folder = OUT / w.name
        fit = read(folder / "fit_summary.json")
        require(fit == value["fit"] and fit["fit_attempts"] == 1 and
                read(folder / "fit_journal.json")["status"] == "saved", "Fit journal mismatch")
        text = (folder / "model.txt").read_text(encoding="utf-8")
        require(digest(folder / "model.txt") == fit["model_sha256"], "Model hash drift")
        model = lgb.Booster(model_str=text)
        validate_model(model, av["training"], text)
        truth = truth_matrix(gen, sites.site_id, w.axis())
        results = {}
        for name, path in (("B3", pred_path(w.name, "B3")), ("P006", folder / "predictions.parquet")):
            if name == "P006":
                require(read(folder / "prediction_manifest.json") == dict(sha256=digest(path), model_sha256=fit["model_sha256"]), "Prediction manifest differs")
            pred = load_prediction(path, truth)
            ss = scores(truth, pred, groups, price, av["cohort_ids"], "fixed_cohort")
            require(ss == value[name + "_scores"], "Stored scores do not reproduce")
            require(diagnostics(truth, pred, sites, price, av["cohort_ids"], av["training"]["categories"]) == value["diagnostics"][name],
                    "Stored diagnostics do not reproduce")
            results[name] = ss
            if name == "P006":
                # Fresh process and freshly rebuilt inputs, all sites on four boundary/day hours.
                axis = w.axis()[[0, 11, 23, len(w.axis())-1]]
                d, x = prediction_frame(w, axis, sites, fa, fb, spec, av["training"])
                fresh = prediction_values(d, model.predict(x))
                stored = pred.stack().reorder_levels([1, 0]).reindex(pd.MultiIndex.from_frame(d[KEY])).to_numpy()
                np.testing.assert_allclose(fresh, stored, rtol=1e-12, atol=0)
        require(gate(results["B3"], results["P006"], w.name in ("T0", "V1")) == value["comparison"], "Stored gate differs")
        del model, truth, pred, d, x
        gc.collect()
        log(f"{w.name}: saved scores, diagnostics, reloaded development probe verified")
    expected = "P006" if all(p["comparison"]["passes"] for p in summary["periods"]) else "B3"
    require(summary["selection"] == expected, "Selection mismatch")
    intact(pre, live=False)
    write_json(OUT / "verification.json", dict(status="passed", at=now(), new_fits=0, models=4,
        predictions=4, score_and_decision_reproduced=True, development_probe_reproduced=True,
        protected_files=len(pre["protected_sha256"]), selection=expected))
    log(f"P006 verification passed; selection={expected}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    require(sys.version_info[:2] == (3, 13) and not sysconfig.get_config_var("Py_GIL_DISABLED"), "Regular CPython 3.13 required")
    {"--precheck": precheck, "--run": execute, "--verify": verify}[sys.argv[1]]()
