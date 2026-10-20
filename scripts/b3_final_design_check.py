"""P1 availability and frozen design only. Never fit, predict or calculate errors."""
from __future__ import annotations

import json
from pathlib import Path
import platform
import sys
import sysconfig

import numpy as np
import pandas as pd

from s0_validation_check import ROOT, SOURCE, INPUTS, baseline_contract, digest, now, write_json
from validation_core import KST, KEY, Window, clock, metadata, require, times, unique, weights

OUT = ROOT / "data/validation/2026-10-20_B3_final_design"
B3 = ROOT / "data/validation/2026-10-20_B3_L1"
FINAL = Window("B3_final", "2025-06-01", "2026-06-01", "2023-06-01", "2025-05-31")
EDA = ROOT / "reports/EDA_진행계획.md"
REPORT = ROOT / "reports/2026-10-20_06_B3_최종학습과_평가절차_설계.md"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    require(sys.version_info[:2] == (3, 13) and not sysconfig.get_config_var("Py_GIL_DISABLED"),
            "Regular CPython 3.13 required")
    require(not (OUT / "summary.json").exists(), "Completed design check exists")
    started = now()
    pre = json.loads((B3 / "precheck.json").read_text(encoding="utf-8"))
    result = json.loads((B3 / "summary.json").read_text(encoding="utf-8"))
    require(result["status"] == "passed" and result["selection"] == "B3_development_pass",
            "B3 must pass its registered development rule")
    require(result["precheck_sha256"] == digest(B3 / "precheck.json"), "B3 preregistration drift")
    expected = dict(n_estimators=600, learning_rate=0.06, num_leaves=127, min_child_samples=200,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                    n_jobs=-1, objective="regression_l1")
    require(result["parameters"] == pre["parameters"] == expected, "B3 parameters changed")
    before = {f: digest(SOURCE / f) for f in INPUTS}
    require(before == pre["source_sha256"], "Supplied files changed")
    protected = dict(pre["protected_sha256"])
    for path in B3.rglob("*"):
        if path.is_file():
            protected[str(path.relative_to(ROOT))] = digest(path)
    require(protected[str(EDA.relative_to(ROOT))] == digest(EDA), "EDA plan changed")
    for rel, expected_hash in protected.items():
        require(digest(ROOT / rel) == expected_hash, f"Prior artifact drift: {rel}")

    spec = baseline_contract()
    weather, common = spec["FEATS"][:15], spec["FEATS"][15:]
    features = [f"{c}_r{r}" for r in (10, 17) for c in weather] + common
    for p in result["periods"]:
        require(features == p["fits"][0]["features"], "Joint feature order changed")
    sites = pd.read_csv(SOURCE / "sites.csv", encoding="utf-8-sig")
    capacities = metadata(sites)
    groups = sites.current_group.map({"A": "G1", "B": "G2"})
    require(groups.notna().all(), "Incomplete reference composition")
    gen = pd.read_parquet(SOURCE / "train/generation.parquet", columns=KEY + ["energy"])
    gen["ts"] = times(gen.ts)
    unique(gen, KEY)
    require(gen.site_id.isin(sites.site_id).all() and np.isfinite(gen.energy).all(), "Invalid truth")
    require(gen.ts.eq(gen.ts.dt.floor("h")).all(), "Nonhourly generation")
    require(gen.ts.max() <= FINAL.axis()[0] - pd.Timedelta(hours=1), "Truth overlaps target period")
    train = FINAL.training(gen)
    require(len(train) > 0, "Empty final training set")
    meta = sites.set_index("site_id")
    daylight = np.zeros(len(train), dtype=bool)
    for pos in range(0, len(train), 250000):
        part = train.iloc[pos:pos+250000]
        locations = meta.loc[part.site_id]
        elev = spec["solar"](times(part.ts), locations.lat.to_numpy(), locations.lon.to_numpy())[0]
        daylight[pos:pos+len(part)] = elev > 0
    counts = train.groupby("site_id").size().reindex(sites.site_id, fill_value=0)
    day_counts = train.loc[daylight].groupby("site_id").size().reindex(sites.site_id, fill_value=0)
    require(counts.gt(0).all() and day_counts.gt(0).all(), "Final model lacks a site's targets")
    train_start = pd.Timestamp(FINAL.train_start, tz=KST)
    before_start = (gen.ts - pd.Timedelta(hours=1)) < train_start
    after_cutoff = gen.ts > pd.Timestamp(FINAL.cutoff, tz=KST)
    require(int(before_start.sum()) + int(after_cutoff.sum()) + len(train) == len(gen),
            "Training partition mismatch")
    former_v2 = train.ts.sub(pd.Timedelta(hours=1)).ge(pd.Timestamp("2025-01-01", tz=KST))

    price = pd.read_csv(SOURCE / "smp.csv", encoding="utf-8-sig")
    price["ts"] = times(price.ts)
    unique(price, ["ts"])
    axis = FINAL.axis()
    require(times(price.ts).equals(axis) and len(axis) == 8760, "Final time grid changed")
    require(np.array_equal(price.k.to_numpy(), weights(axis)), "Evaluation k mismatch")
    require(np.isfinite(price.smp).all() and price.smp.ge(0).all(), "Invalid supplied prices")
    issuance = clock(axis).prediction_at
    cutoff = pd.Timestamp(FINAL.cutoff, tz=KST)
    require(cutoff < issuance.min(), "Final target leakage")
    s0 = json.loads((ROOT / "data/validation/2026-10-19_S0/summary.json").read_text(encoding="utf-8"))
    require(s0["status"] == "passed" and s0["source_sha256"] == before, "S0 source contract changed")
    require(s0["evaluation_grid_rows"] == len(sites)*len(axis), "S0 grid mismatch")
    joins = [r for r in s0["joins"] if r["scope"] == "evaluation_grid"]
    require(len(joins) == 2, "S0 evaluation joins unavailable")

    require(before == {f: digest(SOURCE / f) for f in INPUTS}, "Sources changed during check")
    require(all(digest(ROOT / rel) == h for rel, h in protected.items()), "Previous artifacts changed")
    require("lightgbm" not in sys.modules, "Model library unexpectedly imported")
    summary = dict(status="passed",started_at=started,finished_at=now(),
        runtime=dict(executable=sys.executable,python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__),
        window=vars(FINAL),model_parameters=result["parameters"],features=features,
        source_sha256=before,protected_sha256=protected,design_prereg_sha256=digest(REPORT),
        code_sha256={str(p.relative_to(ROOT)):digest(p) for p in
                     (Path(__file__),ROOT/"scripts/validation_core.py",ROOT/"scripts/s0_validation_check.py")},
        train=dict(rows=len(train),daylight_rows=int(daylight.sum()),sites=int(counts.gt(0).sum()),
            first_ts=str(train.ts.min()),last_ts=str(train.ts.max()),before_start_rows=int(before_start.sum()),
            after_cutoff_rows=int(after_cutoff.sum()),former_V2_rows=int(former_v2.sum()),
            min_site_rows=int(counts.min()),max_site_rows=int(counts.max()),
            min_daylight_site_rows=int(day_counts.min()),max_daylight_site_rows=int(day_counts.max())),
        supplied_truth=dict(rows=len(gen),first_ts=str(gen.ts.min()),last_ts=str(gen.ts.max()),
                            target_period_rows=int(gen.ts.isin(axis).sum())),
        capacity_kw={str(k):float(v) for k,v in capacities.items()},
        group_sites={str(k):int(v) for k,v in sites.current_group.value_counts().items()},
        evaluation=dict(times=len(axis),site_grid_rows=len(axis)*253,group_rows=len(axis)*2,
            first_ts=str(axis[0]),last_ts=str(axis[-1]),first_prediction_at=str(issuance.min()),
            cutoff_gap_hours=(issuance.min()-cutoff).total_seconds()/3600,
            zero_price_times=int(price.smp.eq(0).sum()),s0_joins=joins),
        model_fit_count=0,model_prediction_count=0,model_scores_computed=False,V2_scores_read=False,
        source_unchanged=True,protected_unchanged=True,
        score_unseen_future_holdout_available=False,
        future_execution_budget=dict(final_model_fits=1,alternative_training_windows=0,additional_candidates=0))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT / "preregistered_design.md").write_bytes(REPORT.read_bytes())
    (OUT / "executed_code.py").write_bytes(Path(__file__).read_bytes())
    write_json(OUT / "summary.json", summary)
    print(f"P1 passed: train={len(train):,}, daylight={int(daylight.sum()):,}, sites={int(counts.gt(0).sum())}; fits=0, predictions=0, scores=0",flush=True)


if __name__ == "__main__":
    main()
