"""D1: diagnose frozen V1 B0 outputs. No model fitting, prediction or weight search."""
from __future__ import annotations

import json
import math
from pathlib import Path
import platform
import sys
import sysconfig
import unittest

import numpy as np
import pandas as pd

from s0_validation_check import ROOT, SOURCE, INPUTS, digest, log, now, write_json
from validation_core import KEY, WINDOWS, group_totals, local_score, metadata, require, times, truth_matrix, unique, weights

OUT = ROOT / "data/validation/2026-10-20_D1_V1_residuals"
REPORT = ROOT / "reports/2026-10-20_02_V1_집합잔차와_라운드_진단.md"
B0 = ROOT / "data/validation/2026-10-19_B0_V1"


def safe_ratio(a, b):
    return float(a / b) if b else None


def residual_summary(error, individual_abs, q):
    e, a, q = (np.asarray(v, dtype=float) for v in (error, individual_abs, q))
    require(e.ndim == a.ndim == q.ndim == 1 and len(e) == len(a) == len(q) and len(e) > 0,
            "Invalid diagnostic axes")
    require(np.isfinite(np.stack([e, a, q])).all() and (q >= 0).all(), "Invalid diagnostic values")
    require((a + 1e-7 >= np.abs(e)).all(), "Triangle inequality violated")
    w = float(np.sum(q * np.abs(e)))
    raw = float(np.sum(q * a))
    pos, neg = float(np.sum(q * np.maximum(e, 0))), float(np.sum(q * np.maximum(-e, 0)))
    require(np.isclose(pos + neg, w, rtol=1e-12, atol=1e-5), "Direction sum mismatch")
    contribution = np.sort(q * np.abs(e))[::-1]
    top = {}
    for percent in (1, 5):
        n = math.ceil(len(e) * percent / 100)
        top[f"top_{percent}pct_times"] = n
        top[f"top_{percent}pct_W_share"] = safe_ratio(contribution[:n].sum(), w)
    return dict(times=len(e), W=w, price_k_sum=float(q.sum()), mae_kwh=float(np.abs(e).mean()),
                weighted_mae_kwh=safe_ratio(w, q.sum()), bias_kwh=float(e.mean()),
                weighted_bias_kwh=safe_ratio(np.sum(q*e), q.sum()),
                over_W=pos, under_W=neg, over_W_share=safe_ratio(pos, w),
                individual_abs_weighted=raw, cancellation_ratio=1-w/raw if raw else None, **top)


def round_summary(e10, e17, q):
    e10, e17, q = (np.asarray(v, dtype=float) for v in (e10, e17, q))
    require(e10.shape == e17.shape == q.shape and len(q) > 0, "Round axis mismatch")
    require(np.isfinite(np.stack([e10, e17, q])).all() and (q >= 0).all(), "Invalid round input")
    avg = (e10+e17)/2
    w10, w17, wa = [float(np.sum(q*np.abs(e))) for e in (e10, e17, avg)]
    ref = (w10+w17)/2
    require(wa <= ref + max(1e-5, ref*1e-12), "Ensemble convexity failed")
    corr = float(np.corrcoef(e10, e17)[0, 1]) if np.std(e10) > 0 and np.std(e17) > 0 else None
    active = (np.abs(e10)+np.abs(e17)) > 1e-8
    return dict(times=len(q), W10=w10, W17=w17, Wmean=wa,
                mean_vs_W10_pct=(wa/w10-1)*100 if w10 else None,
                mean_vs_W17_pct=(wa/w17-1)*100 if w17 else None,
                averaging_gain_pct=safe_ratio(ref-wa, ref)*100 if ref else None,
                error_correlation=corr, nonzero_round_error_times=int(active.sum()),
                opposite_sign_fraction=float(np.mean((e10[active]*e17[active]) < 0)) if active.any() else None)


def period_masks(axis):
    start = axis-pd.Timedelta(hours=1)
    masks = [("all", np.ones(len(axis), dtype=bool))]
    masks += [(f"Q{v}", start.quarter == v) for v in sorted(set(start.quarter))]
    masks += [(f"M{v:02}", start.month == v) for v in sorted(set(start.month))]
    masks += [(f"H{v:02}", axis.hour == v) for v in sorted(set(axis.hour))]
    masks += [(f"k={v}", weights(axis) == v) for v in (1, 3)]
    return masks


def frozen_artifacts():
    paths = [ROOT / "reports/EDA_진행계획.md"]
    for folder in (B0, ROOT / "data/validation/2026-10-19_B1_V1", ROOT / "data/validation/2026-10-20_B0_V2"):
        paths += [folder / name for name in ("summary.json", "predictions_mean.parquet", "predictions_round_10.parquet",
                                             "predictions_round_17.parquet", "model_round_10.txt", "model_round_17.txt")]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    require(sys.version_info[:2] == (3, 13) and not sysconfig.get_config_var("Py_GIL_DISABLED"), "Regular CPython 3.13 required")
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_residual_diagnostics.py")
    tests = unittest.TextTestRunner(verbosity=2).run(suite)
    require(tests.wasSuccessful(), "Diagnostic contracts failed")
    require(not (OUT / "summary.json").exists(), "Completed diagnosis exists")
    OUT.mkdir(parents=True, exist_ok=True)
    started = now()
    protected = frozen_artifacts()
    sources = {f: digest(SOURCE / f) for f in INPUTS}
    code = [Path(__file__), ROOT / "scripts/validation_core.py", ROOT / "scripts/s0_validation_check.py",
            ROOT / "tests/test_residual_diagnostics.py"]
    code_hash = {str(p.relative_to(ROOT)): digest(p) for p in code}
    (OUT / "executed_code").mkdir(exist_ok=True)
    for p in code:
        (OUT / "executed_code" / p.name).write_bytes(p.read_bytes())
    b0 = json.loads((B0 / "summary.json").read_text(encoding="utf-8"))
    require(b0["status"] == "passed" and sources == b0["source_sha256"], "B0 source drift")
    sites = pd.read_csv(SOURCE / "sites.csv", encoding="utf-8-sig")
    metadata(sites)
    groups = sites.set_index("site_id").current_group.map({"A": "G1", "B": "G2"})
    gen = pd.read_parquet(SOURCE / "train/generation.parquet")
    gen["ts"] = times(gen.ts)
    truth = truth_matrix(gen, sites.site_id, WINDOWS[0].axis())
    fixed_ids = b0["fixed_148_sites"]
    require(truth[fixed_ids].notna().all().all() and len(fixed_ids) == 148, "Fixed cohort drift")
    price = pd.read_csv(SOURCE / "train/smp.csv", encoding="utf-8-sig")
    price["ts"] = times(price.ts)
    unique(price, ["ts"])
    pred = {}
    for label in ("round_10", "round_17", "mean"):
        frame = pd.read_parquet(B0 / f"predictions_{label}.parquet")
        unique(frame, KEY)
        require(len(frame) == 2222352, "Full V1 prediction grid mismatch")
        pred[label] = frame.pivot(index="ts", columns="site_id", values="kwh").reindex(index=truth.index, columns=truth.columns)
        require(np.isfinite(pred[label].to_numpy()).all() and pred[label].ge(0).all().all(), "Invalid frozen prediction")
    pd.testing.assert_frame_equal((pred["round_10"]+pred["round_17"])/2, pred["mean"])
    group_rows, round_rows, correlation_rows, site_rows, sample_rows = [], [], [], [], []
    for sample, ids in (("fixed_148", fixed_ids), ("common_253", truth.columns.tolist())):
        a, complete = group_totals(truth[ids], groups)
        axis = a.index
        require(len(axis) == (8784 if sample == "fixed_148" else 178), "Registered sample drift")
        q = price.set_index("ts").loc[axis].smp.to_numpy()*weights(axis)
        require(np.isfinite(q).all() and (q >= 0).all(), "Invalid weights")
        totals = {name: group_totals(mat[ids], groups)[0].loc[axis] for name, mat in pred.items()}
        old = next(r for r in b0["group_scores"] if r["sample"] == sample and r["period"] == "all")
        scored = local_score(truth[ids], pred["mean"][ids], groups, price, axis)
        for field in ("W_local", "R_local", "S_local", "price_k_sum", "group_mae_kwh"):
            require(np.isclose(old[field], scored[field], rtol=1e-12, atol=1e-5), "Original B0 score not reproduced")
        sample_rows.append(dict(sample=sample, sites=len(ids), first_ts=str(axis.min()), last_ts=str(axis.max()), **scored))
        for group in ("G1", "G2"):
            members = [i for i in ids if groups[i] == group]
            err = pred["mean"].loc[axis, members]-truth.loc[axis, members]
            e = totals["mean"][group].to_numpy()-a[group].to_numpy()
            np.testing.assert_allclose(err.sum(axis=1).to_numpy(), e, rtol=1e-11, atol=1e-8)
            iabs = err.abs().sum(axis=1).to_numpy()
            for period, mask in period_masks(axis):
                if not mask.any():
                    continue
                group_rows.append(dict(sample=sample, group=group, period=period, sites=len(members),
                                       **residual_summary(e[mask], iabs[mask], q[mask])))
                round_rows.append(dict(sample=sample, group=group, period=period,
                    **round_summary((totals["round_10"][group]-a[group]).to_numpy()[mask],
                                    (totals["round_17"][group]-a[group]).to_numpy()[mask], q[mask])))
            main_row = next(r for r in group_rows if r["sample"] == sample and r["group"] == group and r["period"] == "all")
            for prefix in ("M", "Q", "H", "k="):
                parts = [r for r in group_rows if r["sample"] == sample and r["group"] == group and r["period"].startswith(prefix)]
                require(sum(r["times"] for r in parts) == len(axis), "Diagnostic partition time mismatch")
                require(np.isclose(sum(r["W"] for r in parts), main_row["W"], rtol=1e-12), "Diagnostic W partition mismatch")
            # Coherent sign and site contributions only use the pre-registered k=3 scope.
            high = weights(axis) == 3
            arr = err.to_numpy()[high]
            net = e[high]
            active = np.abs(arr) > 1e-8
            eligible = active.sum(axis=1) > 0
            aligned = ((arr*np.sign(net)[:, None]) > 0).sum(axis=1)
            valid = eligible & (np.abs(net) > 1e-8)
            fraction = aligned[valid]/active[valid].sum(axis=1)
            alignment = safe_ratio(np.sum(fraction*q[high][valid]), q[high][valid].sum())
            cf = err.loc[axis[high]].divide(sites.set_index("site_id").capacity_kw[members], axis=1)
            corr = cf.corr().to_numpy()
            values = corr[np.triu_indices(len(members), 1)]
            values = values[np.isfinite(values)]
            correlation_rows.append(dict(sample=sample, group=group, period="k=3", sites=len(members),
                times=int(high.sum()), finite_pairs=len(values), eligible_sign_times=int(valid.sum()),
                weighted_net_direction_site_fraction=alignment,
                pair_cf_error_corr_mean=float(values.mean()) if len(values) else None,
                pair_cf_error_corr_median=float(np.median(values)) if len(values) else None))
            signed_contribution = (arr*np.sign(net)[:, None]*q[high][:, None]).sum(axis=0)
            require(np.isclose(signed_contribution.sum(), np.sum(np.abs(net)*q[high]), rtol=1e-11, atol=1e-5),
                    "Signed site contributions do not add to group W")
            for j, site in enumerate(members):
                site_rows.append(dict(sample=sample, group=group, site_id=site, period="k=3",
                    W_directional_contribution=float(signed_contribution[j]),
                    individual_weighted_abs=float((np.abs(arr[:, j])*q[high]).sum()),
                    kwh_bias=float(arr[:, j].mean())))
        total_w = sum(r["W"] for r in group_rows if r["sample"] == sample and r["period"] == "all")
        require(np.isclose(total_w, old["W_local"], rtol=1e-12), "Group W differs from local score")
        log(f"{sample}: {len(ids)} sites, {len(axis)} hours; W reproduced; partitions verified")
    for name, rows in (("group_diagnostics", group_rows), ("round_diagnostics", round_rows),
                       ("coherence", correlation_rows), ("site_contributions", site_rows)):
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    require(protected == frozen_artifacts(), "EDA or baseline artifacts changed")
    require(sources == {f: digest(SOURCE / f) for f in INPUTS}, "Source changed")
    require(code_hash == {str(p.relative_to(ROOT)): digest(p) for p in code}, "Code changed")
    # Protected V2 files are hashed as opaque bytes only, never read for diagnosis.
    write_json(OUT / "summary.json", dict(experiment="D1_V1_residuals", status="passed", started_at=started,
        finished_at=now(), runtime=dict(python=platform.python_version(), executable=sys.executable,
        numpy=np.__version__, pandas=pd.__version__), synthetic_tests=tests.testsRun,
        source_sha256=sources, protected_artifact_sha256=protected, code_sha256=code_hash,
        model_fit_count=0, model_prediction_count=0, weight_search_count=0, V2_diagnosed=False,
        samples=sample_rows, groups=group_rows, rounds=round_rows, coherence=correlation_rows))
    log("D1 completed; no fits/predict/weight search; V2 protected; select one future hypothesis after review")


if __name__ == "__main__":
    main()
