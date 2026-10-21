"""EDA-011: diagnose existing predictions only; never load or run a model."""
from __future__ import annotations

import gc
import json
import platform
import sys
import sysconfig
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from s0_validation_check import ROOT, SOURCE, baseline_contract, digest, now, write_json
from validation_core import KEY, KST, Window, clock, grid, metadata, prepare_forecast, require, times, truth_matrix, unique, weights

OUT = ROOT / "data/eda/2026-10-21_forecast_residual"
ROLL = ROOT / "data/validation/2026-10-20_B3_B5_rolling"
ANNUAL = ROOT / "data/validation/2026-10-20_B3_L1"
WEIGHT = ROOT / "data/validation/2026-10-20_B5_B6_loss_weighting"
REPORT = ROOT / "reports/2026-10-21_03_예보조건별_잔차_EDA.md"
WINDOWS = [Window("R3", "2024-07-01", "2024-10-01", "2022-07-01", "2024-06-30"),
           Window("R4", "2024-10-01", "2025-01-01", "2022-10-01", "2024-09-30"),
           Window("T0", "2023-01-01", "2024-01-01", "2022-01-01", "2022-12-31"),
           Window("V1", "2024-01-01", "2025-01-01", "2022-01-01", "2023-12-31")]
FAMILIES = ["cloud17", "rainA17", "rainB17", "revisionA", "revisionB", "provider_cloud", "rain_provider", "rain_revisionA"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def relative(p):
    return str(p.relative_to(ROOT))


def pred_path(w, method):
    if w in ("R3", "R4"):
        return ROLL / w / method / "predictions.parquet"
    return (ANNUAL / w / method / "predictions_mean.parquet" if method == "B3"
            else WEIGHT / w / method / "predictions.parquet")


def cohort_ids(name):
    if name in ("R3", "R4"):
        return next(p["cohort_ids"] for p in read(ROLL / "availability.json")["periods"] if p["window"]["name"] == name)
    p = read(ROOT / "data/validation/2026-10-20_B2_joint/precheck.json")
    return p["V1_cohort_ids"] if name == "V1" else p["T0"]["cohort_ids"]


def inventory():
    """Check historical digests against frozen snapshots, not edited live plans."""
    pre = read(ROLL / "precheck.json")
    paths = {}
    for rel, expected in pre["source_sha256"].items():
        path = SOURCE / rel
        require(digest(path) == expected, f"Source hash mismatch: {rel}")
        paths[relative(path)] = expected
    for rel, expected in pre["protected_sha256"].items():
        path = ROOT / rel
        # Living plans were intentionally updated for this handoff. Historical
        # expectations are retained in the original precheck, not enforced on live plans.
        if path in (ROOT/"reports/EDA_진행계획.md", ROOT/"reports/성능개선_진행계획.md"):
            continue
        require(digest(path) == expected, f"Historical artifact mismatch: {rel}")
        paths[relative(path)] = expected
    for live, frozen, key in ((REPORT, ROLL / "preregistered_report.md", "report_sha256"),
                              (ROOT / "reports/성능개선_진행계획.md", ROLL / "performance_plan.md", "plan_sha256")):
        require(digest(frozen) == pre[key], "Historical frozen scope mismatch")
    for rel, expected in pre["code_sha256"].items():
        require(digest(ROLL / "executed_code" / Path(rel).name) == expected, "Historical executed-code mismatch")
    for p in ROLL.rglob("*"):
        if p.is_file():
            paths[relative(p)] = digest(p)
    for w in WINDOWS:
        for method in ("B3", "B5"):
            p = pred_path(w.name, method)
            if w.name in ("R3", "R4"):
                pm = read(p.parent / "prediction_manifest.json")
                fit = read(p.parent / "fit_summary.json")
                require(digest(p) == pm["sha256"] and pm["model_sha256"] == fit["model_sha256"], "Prediction manifest mismatch")
                require(digest(p.parent / "model.txt") == fit["model_sha256"], "Model bytes changed")
            require(relative(p) in paths, "Prediction not covered by historical hash")
    return paths


def inspect():
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = inventory()
    sites = pd.read_csv(SOURCE / "sites.csv", encoding="utf-8-sig")
    metadata(sites)
    predictions = []
    for w in WINDOWS:
        for method in ("B3", "B5"):
            p = pred_path(w.name, method)
            d = pd.read_parquet(p)
            d["ts"] = times(d.ts)
            unique(d, KEY)
            require(len(d) == len(w.axis()) * 253 and set(d.site_id) == set(sites.site_id), "Prediction grid differs")
            require(set(d.ts) == set(w.axis()) and np.isfinite(d.kwh).all() and d.kwh.ge(0).all(), "Prediction values/time differ")
            predictions.append(dict(window=w.name, method=method, rows=len(d), sites=d.site_id.nunique(),
                                    first=str(d.ts.min()), last=str(d.ts.max()), cohort_sites=len(cohort_ids(w.name)), sha256=digest(p)))
    schemas = {p: str(pq.read_schema(SOURCE / f"train/forecast_{p}.parquet")) for p in ("wa", "wb")}
    write_json(OUT / "inspection.json", dict(at=now(), runtime=platform.python_version(), predictions=predictions, schemas=schemas,
                                               protected_files=len(hashes), protected_sha256=hashes))
    print(json.dumps(dict(predictions=predictions, protected_files=len(hashes)), ensure_ascii=False, indent=2))


def conditions(d):
    """Fixed physical thresholds. All NaN inputs remain explicit missing conditions."""
    def cut(x, bounds, labels):
        return pd.cut(x, bounds, labels=labels, right=True).astype("string").fillna("missing")
    a, b = d.wa_cloud_r17, d.wb_cloud_r17
    c = (a + b) / 2
    result = {"cloud17": cut(c, [-np.inf, 20, 80, np.inf], ["clear<=20", "mixed20-80", "cloudy>80"])}
    for p, label in (("wa", "rainA17"), ("wb", "rainB17")):
        result[label] = cut(d[p + "_rain_r17"], [-np.inf, 0, 1, np.inf], ["dry=0", "light0-1", "wet>1"])
    for p, name in (("wa", "revisionA"), ("wb", "revisionB")):
        x = d[p + "_cloud_r17"] - d[p + "_cloud_r10"]
        result[name] = cut(x, [-np.inf, -20, 20, np.inf], ["down<=-20", "stable(-20,20]", "up>20"])
    result["provider_cloud"] = cut(a-b, [-np.inf, -20, 20, np.inf], ["A_lower<=-20", "agree(-20,20]", "A_higher>20"])
    for name, x, y in (("rain_provider", d.wa_rain_r17, d.wb_rain_r17),
                       ("rain_revisionA", d.wa_rain_r10, d.wa_rain_r17)):
        valid = x.notna() & y.notna()
        result[name] = pd.Series(np.where(x.gt(0), np.where(y.gt(0), "both_wet", "first_only"),
                                                           np.where(y.gt(0), "second_only", "both_dry")), index=d.index).where(valid, "missing")
    return pd.DataFrame(result, index=d.index)


def forecasts(axis):
    result = []
    audit = []
    for p in ("wa", "wb"):
        cols = [p + "_cloud", p + "_rain"] + (["wb_total_liq"] if p == "wb" else [])
        station = p + "_station"
        # Filters prevent V2/evaluation weather rows entering this analysis.
        raw = pd.read_parquet(SOURCE / f"train/forecast_{p}.parquet", columns=[station, "round", "ts"]+cols,
                              filters=[("ts", ">=", axis.min()), ("ts", "<=", axis.max())])
        clean, n = prepare_forecast(raw, station, cols)
        clean = clean.loc[clean.ts.isin(axis)].copy()
        for col in cols:
            valid = np.isfinite(clean[col]) & (clean[col].between(0, 100) if col.endswith("cloud") else clean[col].ge(0))
            audit.append(dict(provider=p, column=col, rows=len(clean), nonhourly_removed=n, missing=int(clean[col].isna().sum()),
                              invalid=int((clean[col].notna() & ~valid).sum()), first=str(clean.ts.min()), last=str(clean.ts.max())))
            clean[col] = clean[col].where(valid)
        for r in (10, 17):
            q = clean.loc[clean["round"].eq(r)].drop(columns="round").rename(columns={c:c+f"_r{r}" for c in cols})
            q[p+f"_present_r{r}"] = True
            result.append((station, q))
    return result, audit


def statistics(d):
    cap = d.capacity_kw.sum()
    sums = d.groupby("date")[["e3", "capacity_kw"]].sum()
    daily = sums.e3 / sums.capacity_kw
    return dict(rows=len(d), days=d.date.nunique(), sites=d.site_id.nunique(), stationsA=d.wa_station.nunique(),
                stationsB=d.wb_station.nunique(), pairs=len(d[["wa_station", "wb_station"]].drop_duplicates()),
                stationA_days=len(d[["wa_station", "date"]].drop_duplicates()),
                stationB_days=len(d[["wb_station", "date"]].drop_duplicates()),
                actual_cf=d.energy.sum()/cap, B3_cf=d.p3.sum()/cap, B5_cf=d.p5.sum()/cap,
                B3_bias=d.e3.sum()/cap, B5_bias=d.e5.sum()/cap,
                B3_site_abs=np.abs(d.e3).sum()/cap, B5_site_abs=np.abs(d.e5).sum()/cap,
                positive_daily_share=daily.gt(0).mean(), small=bool(d.date.nunique()<20 or min(d.wa_station.nunique(),d.wb_station.nunique())<5))


def group_conditions(z):
    """Use capacity weighted forecast summaries for the whole fixed group-time."""
    cols = [c for c in z.columns if c.endswith(("_r10", "_r17")) and ("cloud" in c or "rain" in c)]
    total = z.groupby("ts").capacity_kw.sum()
    out = pd.DataFrame(index=total.index)
    for c in cols:
        validcap = z.capacity_kw.where(z[c].notna(), 0).groupby(z.ts).sum()
        out[c] = (z[c]*z.capacity_kw).groupby(z.ts).sum()/validcap.replace(0,np.nan)
        out[c] = out[c].where(validcap/total >= .95)
    # Rain occurrence = >=50% of valid group capacity; never subtract cross-provider mm amounts.
    for p in ("wa", "wb"):
        for r in (10, 17):
            c = f"{p}_rain_r{r}"
            validcap = z.capacity_kw.where(z[c].notna(),0).groupby(z.ts).sum()
            share = z.capacity_kw.where(z[c].gt(0),0).groupby(z.ts).sum()/validcap.replace(0,np.nan)
            # Preserve separate group mean rain bins; use occurrence proxy only for event families below.
            out[c+"_wetflag"] = share.ge(.5).astype(float).where(validcap/total >= .95)
    labels = conditions(out)
    occurrence = out.copy()
    for c in cols:
        if "rain" in c:
            occurrence[c] = out[c+"_wetflag"]
    event = conditions(occurrence)
    labels[["rain_provider", "rain_revisionA"]] = event[["rain_provider", "rain_revisionA"]]
    eligible = z.capacity_kw.where(z.main,0).groupby(z.ts).sum()/total >= .95
    for family in FAMILIES:
        labels[family] = labels[family].where(eligible, "outside_main")
    return labels


def block_interval(values):
    """Fixed 7-day circular blocks over calendar days; spatial members move together."""
    x = np.asarray(values, dtype=float)
    if len(x) < 20:
        return (None, None)
    rng = np.random.default_rng(20261021)
    starts = rng.integers(0,len(x),size=(1000,int(np.ceil(len(x)/7))))
    ix = (starts[:,:,None]+np.arange(7)) % len(x)
    means = x[ix.reshape(1000,-1)[:,:len(x)]].mean(axis=1)
    return tuple(float(a) for a in np.quantile(means,[.025,.975]))


def matched_contrasts(d, labels, window, cohort, group):
    rows = []
    z = d.loc[d.main].copy()
    z["solar_bin"] = pd.cut(z.elev,[15,30,50,np.inf],right=False).astype("string")
    keys = ["month", "hour", "solar_bin"]
    for family in FAMILIES:
        z["condition"] = labels.loc[z.index,family]
        for label in sorted(set(z.condition)-{"missing"}):
            a = z.loc[z.condition.eq(label)]
            b = z.loc[~z.condition.isin([label,"missing"])]
            if len(a)==0 or len(b)==0:
                continue
            def cell(q):
                s = q.groupby(keys,observed=True).agg(cap=("capacity_kw","sum"), actual=("energy","sum"),
                                                    pred=("p3","sum"), error=("e3","sum"), days=("date","nunique"))
                for col in ("actual","pred","error"):
                    s[col] /= s.cap
                return s.loc[s.days.ge(5)]
            ca, cb = cell(a), cell(b)
            shared = ca.index.intersection(cb.index)
            if not len(shared):
                continue
            diff = ca.loc[shared,["actual","pred","error"]]-cb.loc[shared,["actual","pred","error"]]
            rows.append(dict(window=window,cohort=cohort,group=group,family=family,condition=label,cells=len(shared),
                             delta_actual=float(diff.actual.mean()),delta_pred=float(diff.pred.mean()),delta_bias=float(diff.error.mean()),
                             positive_cell_share=float(diff.error.gt(0).mean())))
    return rows


def run():
    require(not (OUT / "summary.json").exists(), "Completed EDA exists; use --verify, never silently reselect")
    require("EDA-011" in REPORT.read_text(encoding="utf-8"), "Preregistered report required")
    before = inventory()
    require(before == read(OUT / "inspection.json")["protected_sha256"], "Inputs changed after inspection")
    (OUT / "preregistered_report.md").write_bytes(REPORT.read_bytes())
    (OUT / "eda_plan.md").write_bytes((ROOT / "reports/EDA_진행계획.md").read_bytes())
    (OUT / "executed_code").mkdir(exist_ok=True)
    for p in (Path(__file__),ROOT/"scripts/s0_validation_check.py",ROOT/"scripts/validation_core.py"):
        (OUT/"executed_code"/p.name).write_bytes(p.read_bytes())
    write_json(OUT/"precheck.json",dict(at=now(),protected_sha256=before,report_sha256=digest(REPORT),
                                       code_sha256=digest(Path(__file__)),model_fit_count=0,inference_count=0))
    sites = pd.read_csv(SOURCE/"sites.csv",encoding="utf-8-sig"); metadata(sites)
    fixed = set(cohort_ids("V1"))
    spec = baseline_contract()
    allrows, group_rows, contributions, matched, audits, window_info, blocks, correlations = [], [], [], [], [], [], [], []
    for w in WINDOWS:
        print(f"[{now()}] {w.name}: existing predictions and forecast joins",flush=True)
        axis = w.axis(); ids = cohort_ids(w.name)
        gen = pd.read_parquet(SOURCE/"train/generation.parquet",columns=KEY+["energy"],filters=[("ts",">=",axis.min()),("ts","<=",axis.max())])
        gen["ts"] = times(gen.ts); unique(gen,KEY)
        truth = truth_matrix(gen,ids,axis)
        require(truth.notna().all().all(), "Fixed cohort incomplete")
        if w.name in ("R3","R4"):
            fulltruth = truth_matrix(gen,sites.site_id,axis)
            require(set(fulltruth.columns[fulltruth.notna().all()])==set(ids), "Quarter cohort changed")
        d = grid(sites.loc[sites.site_id.isin(ids)],axis)
        d = d.merge(gen[KEY+["energy"]],on=KEY,how="left",validate="one_to_one",sort=False)
        for method in ("B3","B5"):
            p = pd.read_parquet(pred_path(w.name,method)); p["ts"] = times(p.ts);unique(p,KEY)
            require(len(p)==len(axis)*253 and set(p.ts)==set(axis) and set(p.site_id)==set(sites.site_id),"Prediction keys changed")
            d = d.merge(p.rename(columns={"kwh":"p3" if method=="B3" else "p5"}),on=KEY,how="left",validate="one_to_one",sort=False)
        require(len(d)==len(axis)*len(ids) and d[["energy","p3","p5"]].notna().all().all(), "Truth/prediction join changed")
        fcs, qa = forecasts(axis)
        audits.extend([dict(window=w.name,**a) for a in qa])
        for station, q in fcs:
            d = d.merge(q,on=[station,"ts"],how="left",validate="many_to_one",sort=False)
        require(len(d)==len(axis)*len(ids), "Weather join expanded")
        for pfx in ("wa", "wb"):
            for r in (10,17):
                present=d[f"{pfx}_present_r{r}"].eq(True)
                audits.append(dict(window=w.name,provider=pfx,round=r,joined_rows=len(d),
                                   unmatched_rows=int((~present).sum()),
                                   matched_cloud_nan=int((present & d[f"{pfx}_cloud_r{r}"].isna()).sum()),
                                   matched_rain_nan=int((present & d[f"{pfx}_rain_r{r}"].isna()).sum())))
        c = clock(d.ts)
        d["date"] = c.target_day.to_numpy(); d["month"] = c.start.dt.month.to_numpy(); d["hour"] = c.start.dt.hour.to_numpy()
        d["elev"] = spec["solar"](times(d.ts),d.lat.to_numpy(),d.lon.to_numpy())[0]
        d["main"] = d.hour.between(9,16) & d.elev.ge(15)
        d["group"] = d.current_group.map({"A":"G1","B":"G2"})
        d["e3"] = d.p3-d.energy; d["e5"] = d.p5-d.energy
        d["membership"] = np.where(d.site_id.isin(fixed),"fixed_148","added_complete")
        d["station_pair"] = d.wa_station.astype(str)+"/"+d.wb_station.astype(str)
        labels = conditions(d)
        price = pd.read_csv(SOURCE/"train/smp.csv",encoding="utf-8-sig")
        price["ts"] = times(price.ts);unique(price,["ts"])
        pp = price.set_index("ts").smp.reindex(axis)
        require(np.isfinite(pp).all() and pp.ge(0).all(), "Price missing")
        d["q"] = d.ts.map(pp)*weights(d.ts)
        # Full-group sign determines additive subgroup/condition contributions.
        for method, err in (("B3","e3"),("B5","e5")):
            full_e = d.groupby(["group","ts"])[err].transform("sum")
            d[method+"_contribution"] = d.q*np.sign(full_e)*d[err]
        for cohort, subset in (("complete",d),("fixed_148",d.loc[d.membership.eq("fixed_148")]),
                               ("added_complete",d.loc[d.membership.eq("added_complete")])):
            if len(subset)==0:
                continue
            for group in ("G1","G2"):
                z = subset.loc[subset.group.eq(group)]
                if len(z)==0:
                    continue
                day = z.loc[z.main]
                for family in FAMILIES:
                    for label, ix in labels.loc[day.index].groupby(family).groups.items():
                        allrows.append(dict(window=w.name,cohort=cohort,group=group,family=family,condition=label,**statistics(day.loc[ix])))
                matched.extend(matched_contrasts(z,labels,w.name,cohort,group))
                group_label = group_conditions(z)
                a = z.groupby("ts")[["energy","p3","p5","capacity_kw"]].sum()
                a["date"] = clock(a.index).target_day.to_numpy()
                a["q"] = pp*weights(axis)
                for method, pred in (("B3","p3"),("B5","p5")):
                    a[method+"_e"] = a[pred]-a.energy
                    a[method+"_W"] = a.q*np.abs(a[method+"_e"])
                for family in FAMILIES:
                    subtotal = []
                    for label, ix in group_label.groupby(family).groups.items():
                        b = a.loc[ix]
                        r = dict(window=w.name,cohort=cohort,group=group,family=family,condition=label,times=len(b),days=b.date.nunique(),
                                 actual_cf=b.energy.sum()/b.capacity_kw.sum(), B3_cf=b.p3.sum()/b.capacity_kw.sum(), B5_cf=b.p5.sum()/b.capacity_kw.sum())
                        for method in ("B3","B5"):
                            e = b[method+"_e"]
                            r[method+"_W"] = b[method+"_W"].sum()
                            r[method+"_over_W"] = (b.q*e.clip(lower=0)).sum()
                            r[method+"_under_W"] = (b.q*(-e).clip(lower=0)).sum()
                            r[method+"_bias"] = e.sum()/b.capacity_kw.sum()
                        r["B5_change_pct"] = (r["B5_W"]/r["B3_W"]-1)*100 if r["B3_W"] else None
                        group_rows.append(r);subtotal.append(r)
                    for method in ("B3","B5"):
                        require(np.isclose(sum(r[method+"_W"] for r in subtotal),a[method+"_W"].sum(),rtol=1e-12), "Group condition W not conserved")
                # Daily signs and uncertainty for B3 bias, using complete group-time labels.
                for family in FAMILIES:
                    for label in sorted(set(group_label[family]) - {"missing","outside_main"}):
                        mask = group_label[family].eq(label)
                        num = a["B3_e"].where(mask,0).groupby(a.date).sum()
                        den = a.capacity_kw.where(mask,0).groupby(a.date).sum()
                        active = den.gt(0)
                        daily = (num/den.replace(0,np.nan)).fillna(0)
                        # Zero on inactive dates estimates calendar-day signed contribution, not conditional mean.
                        low, high = block_interval(daily.to_numpy())
                        blocks.append(dict(window=w.name,cohort=cohort,group=group,family=family,condition=label,
                                           active_days=int(active.sum()),calendar_days=len(daily),positive_active_share=float((daily.loc[active]>0).mean()),
                                           calendar_bias_mean=float(daily.mean()),block_low=low,block_high=high,small=bool(active.sum()<20)))
                # Monthly/hour/k summary: no alternative thresholds are searched.
                for dimension, series in (("month",clock(a.index).start.dt.month.to_numpy()),
                                          ("hour",clock(a.index).start.dt.hour.to_numpy()),("k",weights(a.index))):
                    for val in sorted(set(series)):
                        b = a.loc[series==val]
                        audits.append(dict(window=w.name,cohort=cohort,group=group,dimension=dimension,value=int(val),times=len(b),
                                           B3_W=float(b.B3_W.sum()),B5_W=float(b.B5_W.sum())))
        # Preserve full W exactly while assigning signed contributions to site weather/cohort conditions.
        for family in FAMILIES:
            cond = labels[family].where(d.main,"outside_main")
            q = d.assign(condition=cond)
            cr = q.groupby(["group","membership","condition"])[["B3_contribution","B5_contribution"]].sum().reset_index()
            contributions.extend([dict(window=w.name,family=family,**r) for r in cr.to_dict("records")])
            for method, err in (("B3","e3"),("B5","e5")):
                direct = d.groupby(["group","ts"]).agg(e=(err,"sum"),q=("q","first"))
                require(np.isclose(cr[method+"_contribution"].sum(),(direct.q*direct.e.abs()).sum(),rtol=1e-12),"Signed contribution not conserved")
        # Simple versus within-month/hour/solar correlations on unique weather-pair/time records.
        day = d.loc[d.main].copy()
        pair = day.groupby(["station_pair","ts","month","hour"],observed=True).agg(
            cap=("capacity_kw","sum"),energy=("energy","sum"),e3=("e3","sum"),cloud=("wa_cloud_r17","first"),rain=("wa_rain_r17","first"))
        pair["actual_cf"]=pair.energy/pair.cap;pair["bias"]=pair.e3/pair.cap
        for col in ("cloud","rain"):
            for target in ("actual_cf","bias"):
                vals=pair[[col,target]].dropna()
                centered=vals-vals.groupby(level=["month","hour"]).transform("mean")
                correlations.append(dict(window=w.name,variable=col,target=target,pair_times=len(vals),
                                         pearson=float(vals[col].corr(vals[target])),within_month_hour=float(centered[col].corr(centered[target]))))
        # Reproduce previous whole-period W without respecifying cohorts.
        direct = d.groupby(["group","ts"]).agg(e3=("e3","sum"),e5=("e5","sum"),q=("q","first"))
        w3=float((direct.q*direct.e3.abs()).sum());w5=float((direct.q*direct.e5.abs()).sum())
        summary=read(ROLL/"summary.json") if w.name in ("R3","R4") else read(ANNUAL/"summary.json")
        period=next(p for p in summary["periods"] if p["window"]["name"]==w.name)
        scorelist=period["methods"]["B3"]["scores"] if w.name in ("R3","R4") else period["B3_scores"]
        expected=next(r["W_local"] for r in scorelist if r["period"]=="all" and r["sample"]!="common_253")
        require(np.isclose(w3,expected,rtol=1e-12),"Historical B3 W mismatch")
        if w.name in ("R3","R4"):
            expected5=next(r["W_local"] for r in period["methods"]["B5"]["scores"] if r["period"]=="all" and r["sample"]=="complete_quarter")
        else:
            p5=next(p for p in read(WEIGHT/"summary.json")["periods"] if p["window"]["name"]==w.name)
            expected5=next(r["W_local"] for r in p5["candidates"]["B5"]["scores"] if r["period"]=="all" and r["sample"]!="common_253")
        require(np.isclose(w5,expected5,rtol=1e-12),"Historical B5 W mismatch")
        window_info.append(dict(window=w.name,sites=len(ids),rows=len(d),main_rows=int(d.main.sum()),
                                first=str(axis.min()),last=str(axis.max()),B3_W=w3,B5_W=w5,
                                membership=d.groupby("membership").site_id.nunique().to_dict(),
                                weather_missing={c:int(d[c].isna().sum()) for c in d if c.endswith(("_r10","_r17"))}))
        del d,gen,truth,labels,day,pair,z,subset,fcs;gc.collect()
    for name,rows in (("site_conditions",allrows),("group_conditions",group_rows),("signed_contributions",contributions),
                      ("matched_contrasts",matched),("audits",audits),("daily_blocks",blocks),("correlations",correlations)):
        pd.DataFrame(rows).to_csv(OUT/(name+".csv"),index=False,encoding="utf-8-sig")
    require(inventory()==before,"Protected artifacts changed during EDA")
    write_json(OUT/"summary.json",dict(status="passed",finished_at=now(),runtime=platform.python_version(),
                                      windows=window_info,model_fit_count=0,inference_count=0,correction_fit_count=0,
                                      V2_analyzed=False,evaluation_analyzed=False,protected_unchanged=True,
                                      output_rows={"site_conditions":len(allrows),"group_conditions":len(group_rows),
                                                   "matched_contrasts":len(matched),"daily_blocks":len(blocks)},
                                      source_code_sha256=digest(Path(__file__))))
    print(json.dumps(dict(status="passed",windows=window_info),ensure_ascii=False,indent=2))


def verify():
    s=read(OUT/"summary.json"); pre=read(OUT/"precheck.json")
    require(s["status"]=="passed" and s["model_fit_count"]==s["inference_count"]==0,"Unexpected execution")
    require(pre["protected_sha256"]==inventory(),"Preservation failed")
    require(pre["code_sha256"]==digest(Path(__file__))==s["source_code_sha256"],"Executed code changed")
    require(digest(OUT/"preregistered_report.md")==pre["report_sha256"],"Frozen report changed")
    g=pd.read_csv(OUT/"group_conditions.csv",encoding="utf-8-sig")
    c=pd.read_csv(OUT/"signed_contributions.csv",encoding="utf-8-sig")
    for w in s["windows"]:
        for family in FAMILIES:
            z=g.loc[g.window.eq(w["window"]) & g.cohort.eq("complete") & g.family.eq(family)]
            require(z.times.sum()==2*len(next(a for a in WINDOWS if a.name==w["window"]).axis()),"Condition time partition differs")
            for method in ("B3","B5"):
                require(np.isclose(z[method+"_W"].sum(),w[method+"_W"],rtol=1e-12),"Full W partition failed")
                require(np.allclose(z[method+"_W"],z[method+"_over_W"]+z[method+"_under_W"],rtol=1e-12),"Direction split failed")
                x=c.loc[c.window.eq(w["window"]) & c.family.eq(family),method+"_contribution"].sum()
                require(np.isclose(x,w[method+"_W"],rtol=1e-12),"Signed W sum failed")
    write_json(OUT/"verification.json",dict(at=now(),status="passed",protected_files=len(pre["protected_sha256"]),
                                           windows=4,families=8,full_W_partitions=64,signed_partitions=64,
                                           historical_B3_B5_W_reproduced=True,code_snapshot_matches=True,
                                           output_sha256={p.name:digest(p) for p in OUT.glob("*.csv")}))
    print("EDA-011 verification passed: 4 windows, 8 families, 64 W and 64 signed partitions; no modeling")


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    require(sys.version_info[:2]==(3,13) and not sysconfig.get_config_var("Py_GIL_DISABLED"),"Regular CPython 3.13 required")
    {"--inspect":inspect,"--run":run,"--verify":verify}[sys.argv[1]]()
