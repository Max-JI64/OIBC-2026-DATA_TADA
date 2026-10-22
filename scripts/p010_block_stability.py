"""Paired saved-prediction block diagnostics; no fitting or model inference."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import sysconfig
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd

from h1_cloud_interaction import WINDOWS, inputs, load_prediction
from forecast_residual_eda import cohort_ids, pred_path
from p006_site_category import intact
from s0_validation_check import ROOT, digest, log, now, write_json
from validation_core import require, truth_matrix, weights

OUT = ROOT / "data/validation/2026-10-22_P010_block_stability"
DEV = ROOT / "data/validation/2026-10-22_P006_site_category"
REPORT = ROOT / "reports/2026-10-22_06_P010_저장예측_블록안정성.md"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def direction(before, after):
    if np.isclose(before, after, rtol=1e-12, atol=0):
        return 0
    return -1 if after < before else 1


def block_summary(axis, start, before, after, days):
    require(days in (7, 14) and len(axis) == len(before) == len(after), "Invalid block scope")
    require(np.isfinite(before).all() and np.isfinite(after).all() and (before >= 0).all() and (after >= 0).all(), "Bad costs")
    target = axis - pd.Timedelta(hours=1)
    anchor = pd.Timestamp(start, tz=axis.tz)
    index = ((target.normalize() - anchor) / pd.Timedelta(days=1)).astype(int) // days
    require((index >= 0).all(), "Bad block anchor")
    total_b, total_p = float(before.sum()), float(after.sum())
    rows = []
    for key in sorted(set(index)):
        mask = index == key
        b, p = float(before[mask].sum()), float(after[mask].sum())
        rb, rp = total_b-b, total_p-p
        rows.append(dict(block=int(key), first_target=str(target[mask].min()), last_target=str(target[mask].max()),
            days=int(len(set(target[mask].date))), hours=int(mask.sum()), B3_W=b, P006_W=p, delta_W=p-b,
            remaining_B3_W=rb, remaining_P006_W=rp, remaining_delta_W=rp-rb,
            remaining_change_pct=(rp/rb-1)*100 if rb else None, remaining_direction=direction(rb, rp)))
    for field, total in (("B3_W", total_b), ("P006_W", total_p)):
        require(np.isclose(sum(r[field] for r in rows), total, rtol=1e-12), "Block partition differs")
    return dict(block_days=days, blocks=len(rows), worsening_blocks=sum(direction(r["B3_W"], r["P006_W"]) > 0 for r in rows),
        improving_blocks=sum(direction(r["B3_W"], r["P006_W"]) < 0 for r in rows),
        removal_reversal_count=sum(r["remaining_direction"] > 0 for r in rows),
        removal_no_strict_improvement_count=sum(r["remaining_direction"] >= 0 for r in rows),
        rows=rows)


def compute():
    pre = read(DEV / "precheck.json")
    intact(pre, live=False)
    sites, gen, price = inputs()
    groups = sites.set_index("site_id").current_group.map({"A": "G1", "B": "G2"})
    output = []
    for w in (w for w in WINDOWS if w.name in ("R3", "R4")):
        result = read(DEV / w.name / "result.json")
        truth = truth_matrix(gen, sites.site_id, w.axis())
        ids = cohort_ids(w.name)
        require(len(ids) == (224 if w.name == "R3" else 252) and truth[ids].notna().all().all(), "Fixed sample differs")
        ppath = DEV / w.name / "predictions.parquet"
        require(digest(ppath) == read(DEV / w.name / "prediction_manifest.json")["sha256"], "Candidate prediction drift")
        pred = {"B3": load_prediction(pred_path(w.name, "B3"), truth), "P006": load_prediction(ppath, truth)}
        q = price.set_index("ts").smp.reindex(w.axis()).to_numpy() * weights(w.axis())
        require(np.isfinite(q).all() and (q >= 0).all(), "Bad price weights")
        costs, breakdown = {}, []
        for name in ("B3", "P006"):
            by_group = []
            for group in ("G1", "G2"):
                members = [i for i in ids if groups[i] == group]
                e = pred[name][members].sum(axis=1).to_numpy() - truth[members].sum(axis=1).to_numpy()
                cost = q * np.abs(e)
                # Independent residual summation, with subtraction order allowed tiny roundoff.
                e2 = (pred[name][members].to_numpy() - truth[members].to_numpy()).sum(axis=1)
                np.testing.assert_allclose(e, e2, rtol=1e-10, atol=1e-8)
                np.testing.assert_allclose(cost.sum(), np.sum(q*np.abs(e2)), rtol=1e-12, atol=1e-5)
                by_group.append(cost)
                for k in (1, 3):
                    breakdown.append(dict(method=name, group=group, k=k, W=float(cost[weights(w.axis()) == k].sum())))
            costs[name] = np.sum(by_group, axis=0)
            expected = next(r["W_local"] for r in result[name + "_scores"] if r["sample"] == "fixed_cohort" and r["period"] == "all")
            require(np.isclose(costs[name].sum(), expected, rtol=1e-12, atol=1e-5), "Stored primary W mismatch")
            require(np.isclose(sum(r["W"] for r in breakdown if r["method"] == name), expected, rtol=1e-12), "Group/k partition differs")
        b, p = costs["B3"], costs["P006"]
        output.append(dict(window=vars(w), cohort_sites=len(ids), hours=len(b), B3_W=float(b.sum()), P006_W=float(p.sum()),
            delta_W=float(p.sum()-b.sum()), change_pct=float((p.sum()/b.sum()-1)*100), breakdown=breakdown,
            block_summaries=[block_summary(w.axis(), w.start, b, p, days) for days in (7, 14)]))
        log(f"P010 {w.name}: paired block diagnostics complete, no model calls")
    intact(pre, live=False)
    return output


def forbidden(*args, **kwargs):
    raise AssertionError("P010 must not fit or infer a model")


def main(verify=False):
    with patch.object(lgb.LGBMRegressor, "fit", forbidden), patch.object(lgb.Booster, "predict", forbidden), patch.object(lgb, "train", forbidden):
        if verify:
            summary = read(OUT / "summary.json")
            require(summary["code_sha256"] == digest(Path(__file__)) and summary["preregistered_sha256"] == digest(OUT / "preregistered_report.md"), "P010 scope/code drift")
            require(compute() == summary["periods"], "P010 results not reproducible")
            write_json(OUT / "verification.json", dict(status="passed", at=now(), new_fits=0, new_inference_calls=0,
                result_reproduced=True, summary_sha256=digest(OUT / "summary.json")))
        else:
            require(not (OUT / "summary.json").exists(), "P010 already complete; use --verify")
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "preregistered_report.md").write_bytes(REPORT.read_bytes())
            (OUT / "executed_code.py").write_bytes(Path(__file__).read_bytes())
            output = compute()
            write_json(OUT / "summary.json", dict(status="passed", at=now(), periods=output,
                code_sha256=digest(Path(__file__)), preregistered_sha256=digest(OUT / "preregistered_report.md"),
                model_fit_count=0, model_inference_calls=0, original_selection="P006", selection_unchanged=True,
                V2_executed=False, evaluation_executed=False))
    log("P010 saved verification passed" if verify else "P010 passed; development selection unchanged")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    require(sys.version_info[:2] == (3, 13) and not sysconfig.get_config_var("Py_GIL_DISABLED"), "Regular CPython 3.13 required")
    main("--verify" in sys.argv)
