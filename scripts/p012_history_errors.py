"""P-012: fixed saved-prediction history diagnostics; no fitting or inference."""
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
from s0_validation_check import ROOT, digest, now, write_json, log
from validation_core import require, truth_matrix, weights

OUT = ROOT / "data/validation/2026-10-22_P012_history_errors"
DEV = ROOT / "data/validation/2026-10-22_P006_site_category"
FINAL = ROOT / "data/final/2026-10-22_P006"
REPORT = ROOT / "reports/2026-10-22_09_P012_학습이력별_오류점검.md"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def allocate(bs, bl, ps, pl, q):
    """Average two replacement orders; allocations sum to full aggregate cost change."""
    arrays = [np.asarray(a, dtype="float64") for a in (bs, bl, ps, pl, q)]
    require(len({a.shape for a in arrays}) == 1 and all(np.isfinite(a).all() for a in arrays), "Bad residual arrays")
    bs, bl, ps, pl, q = arrays
    require((q >= 0).all(), "Bad cost weights")
    bb, pb, bp, pp = [q*np.abs(a+b) for a, b in ((bs, bl), (ps, bl), (bs, pl), (ps, pl))]
    short = ((pb-bb)+(pp-bp))/2
    longer = ((bp-bb)+(pp-pb))/2
    np.testing.assert_allclose(short+longer, pp-bb, rtol=1e-10, atol=1e-6)
    return dict(short=short, longer=longer, before=bb, after=pp)


def evidence(row):
    s, l = row["short"], row["longer"]
    checks = dict(support=s["sites"] >= 10 and l["sites"] >= 10,
        mean_worse=s["sites"] > 0 and s["mean_delta_cf_mae"] > 1e-12,
        median_worse=s["sites"] > 0 and s["median_delta_cf_mae"] > 1e-12,
        majority_worse=s["sites"] > 0 and s["worsening_share"] > .5,
        worse_than_longer=s["sites"] > 0 and l["sites"] > 0 and s["mean_delta_cf_mae"] > l["mean_delta_cf_mae"]+1e-12,
        allocated_W_worse=row["short_allocated_delta_W"] > 1e-5)
    return dict(checks=checks, passes=all(checks.values()))


def protected():
    intact(read(DEV / "precheck.json"), live=False)
    summary, verify = read(FINAL / "summary.json"), read(FINAL / "verification.json")
    require(summary["status"] == verify["status"] == "passed" and verify["summary_sha256"] == digest(FINAL / "summary.json"), "Final verification changed")
    paths = [FINAL / n for n in summary["output_sha256"]] + [FINAL / "summary.json", FINAL / "verification.json"]
    for name, h in summary["output_sha256"].items():
        require(digest(FINAL / name) == h, "Final output drift")
    paths += [DEV / "summary.json", DEV / "precheck.json", DEV / "verification.json"]
    for w in ("R3", "R4"):
        p = DEV / w / "predictions.parquet"
        require(digest(p) == read(DEV / w / "prediction_manifest.json")["sha256"], "Candidate predictions changed")
        paths += [p, DEV / w / "result.json", DEV / w / "fit_summary.json"]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def compute():
    sites, gen, price = inputs()
    site_meta = sites.set_index("site_id")
    rows, per_site, periods = [], [], []
    for w in (w for w in WINDOWS if w.name in ("R3", "R4")):
        result = read(DEV / w.name / "result.json")
        ids = cohort_ids(w.name)
        truth = truth_matrix(gen, sites.site_id, w.axis())
        require(truth[ids].notna().all().all(), "Incomplete fixed truth")
        require(set(ids) <= set(result["fit"]["training"]["categories"]), "Unseen primary ID")
        hist = pd.DataFrame(result["fit"]["training"]["per_site"]).set_index("site_id").loc[ids]
        pred = {"B3": load_prediction(pred_path(w.name, "B3"), truth)[ids],
                "P006": load_prediction(DEV / w.name / "predictions.parquet", truth)[ids]}
        target = truth[ids].to_numpy()
        errors = {name: p.to_numpy()-target for name, p in pred.items()}
        cap = site_meta.loc[ids, "capacity_kw"].to_numpy(dtype="float64")
        require(np.isfinite(cap).all() and (cap > 0).all(), "Bad capacity")
        cf_mae = {name: np.mean(np.abs(e)/cap, axis=0) for name, e in errors.items()}
        kwh_mae = {name: np.mean(np.abs(e), axis=0) for name, e in errors.items()}
        # Recompute via separate pandas expression before group summaries.
        for name in pred:
            direct = (pred[name]-truth[ids]).abs().div(pd.Series(cap, index=ids), axis=1).mean(axis=0).to_numpy()
            np.testing.assert_allclose(cf_mae[name], direct, rtol=1e-12, atol=0)
        delta = cf_mae["P006"]-cf_mae["B3"]
        q = price.set_index("ts").smp.reindex(w.axis()).to_numpy()*weights(w.axis())
        group_labels = site_meta.loc[ids, "current_group"].to_numpy()
        require(np.isfinite(q).all() and (q >= 0).all(), "Bad weights")
        totals = {}
        for name, e in errors.items():
            total = sum(float(np.sum(q*np.abs(e[:, group_labels == g].sum(axis=1)))) for g in ("A", "B"))
            expected = next(r["W_local"] for r in result[name+"_scores"] if r["sample"] == "fixed_cohort" and r["period"] == "all")
            require(np.isclose(total, expected, rtol=1e-12, atol=1e-5), "Stored whole W mismatch")
            totals[name] = total
        month_mask, count_mask = hist.months.to_numpy() <= 6, hist.daylight_rows.to_numpy() <= 2000
        periods.append(dict(window=w.name, sites=len(ids), hours=len(w.axis()), scores=totals,
                            definition_agreement=float(np.mean(month_mask == count_mask))))
        for j, site in enumerate(ids):
            per_site.append(dict(window=w.name, site_id=site, months=int(hist.iloc[j].months), rows=int(hist.iloc[j].daylight_rows),
                B3_cf_mae=float(cf_mae["B3"][j]), P006_cf_mae=float(cf_mae["P006"][j]), delta_cf_mae=float(delta[j])))

        def metrics(mask):
            n = int(mask.sum())
            if not n:
                return dict(sites=0, B3_cf_mae=None, P006_cf_mae=None, mean_delta_cf_mae=None,
                            median_delta_cf_mae=None, worsening_share=None, B3_kwh_mae=None, P006_kwh_mae=None)
            return dict(sites=n, B3_cf_mae=float(cf_mae["B3"][mask].mean()), P006_cf_mae=float(cf_mae["P006"][mask].mean()),
                mean_delta_cf_mae=float(delta[mask].mean()), median_delta_cf_mae=float(np.median(delta[mask])),
                worsening_share=float(np.mean(delta[mask] > 1e-12)),
                B3_kwh_mae=float(kwh_mae["B3"][mask].mean()), P006_kwh_mae=float(kwh_mae["P006"][mask].mean()))

        for definition, short in (("months_le_6", month_mask), ("daylight_rows_le_2000", count_mask)):
            breakdown = []
            for group in ("A", "B"):
                select = group_labels == group
                bs, bl = [errors["B3"][:, select & mask].sum(axis=1) for mask in (short, ~short)]
                ps, pl = [errors["P006"][:, select & mask].sum(axis=1) for mask in (short, ~short)]
                a = allocate(bs, bl, ps, pl, q)
                for k in (1, 3):
                    mask = weights(w.axis()) == k
                    breakdown.append(dict(group=group, k=k, short_delta_W=float(a["short"][mask].sum()),
                        longer_delta_W=float(a["longer"][mask].sum()), B3_W=float(a["before"][mask].sum()),
                        P006_W=float(a["after"][mask].sum())))
            for name in ("B3", "P006"):
                require(np.isclose(sum(z[name+"_W"] for z in breakdown), totals[name], rtol=1e-12, atol=1e-5), "Allocation sample differs")
            s_delta, l_delta = [sum(z[key] for z in breakdown) for key in ("short_delta_W", "longer_delta_W")]
            require(np.isclose(s_delta+l_delta, totals["P006"]-totals["B3"], rtol=1e-10, atol=1e-5), "Allocation total differs")
            row = dict(window=w.name, definition=definition, short=metrics(short), longer=metrics(~short),
                       short_allocated_delta_W=s_delta, longer_allocated_delta_W=l_delta, breakdown=breakdown)
            row["evidence"] = evidence(row)
            rows.append(row)
        log(f"P012 {w.name}: history diagnostics complete; no model calls")
    require(len(rows) == 4, "Diagnostic scope changed")
    return dict(periods=periods, comparisons=rows, per_site=per_site,
        regularization_evidence=all(r["evidence"]["passes"] for r in rows),
        interpretation="Descriptive development diagnostics, no causal overfit claim")


def forbidden(*args, **kwargs):
    raise AssertionError("P012 cannot fit or infer models")


def main(verify=False):
    before = protected()
    with patch.object(lgb.LGBMRegressor, "fit", forbidden), patch.object(lgb, "train", forbidden), patch.object(lgb.Booster, "predict", forbidden):
        if verify:
            summary = read(OUT / "summary.json")
            require(summary["code_sha256"] == digest(Path(__file__)) and
                    summary["scope_sha256"] == digest(OUT / "preregistered_report.md"), "P012 code/scope drift")
            require(summary["protected_sha256"] == before, "P012 protected drift")
            require(compute() == summary["result"], "P012 result not reproduced")
            require(protected() == before, "Artifacts changed")
            write_json(OUT / "verification.json", dict(status="passed", at=now(), result_reproduced=True,
                model_fits=0, model_inference_calls=0, summary_sha256=digest(OUT / "summary.json")))
        else:
            OUT.mkdir(parents=True, exist_ok=True)
            require(not (OUT / "summary.json").exists(), "P012 already complete; use --verify")
            (OUT / "preregistered_report.md").write_bytes(REPORT.read_bytes())
            (OUT / "executed_code.py").write_bytes(Path(__file__).read_bytes())
            result = compute()
            require(protected() == before, "Artifacts changed")
            write_json(OUT / "summary.json", dict(status="passed", at=now(), result=result,
                protected_sha256=before, code_sha256=digest(Path(__file__)), scope_sha256=digest(REPORT),
                model_fits=0, model_inference_calls=0, V2_executed=False, evaluation_executed=False))
    log("P012 separate verification passed" if verify else "P012 analysis complete")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    require(sys.version_info[:2] == (3, 13) and not sysconfig.get_config_var("Py_GIL_DISABLED"), "Regular CPython 3.13 required")
    main("--verify" in sys.argv)
