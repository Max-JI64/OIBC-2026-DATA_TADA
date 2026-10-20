"""D3: frozen V1 B2/B3 errors only, without fitting, inference or correction."""
from __future__ import annotations

import json
from pathlib import Path
import platform
import sys
import sysconfig
import unittest

import numpy as np
import pandas as pd

from s0_validation_check import ROOT, SOURCE, INPUTS, digest, log, now, write_json
from validation_core import KEY, WINDOWS, group_totals, local_score, metadata, require, times, truth_matrix, unique, weights

OUT = ROOT / "data/validation/2026-10-20_D3_B3_summer"
B2 = ROOT / "data/validation/2026-10-20_B2_joint"
B3 = ROOT / "data/validation/2026-10-20_B3_L1"
P1 = ROOT / "data/validation/2026-10-20_B3_final_design"
REPORT = ROOT / "reports/2026-10-20_07_B3_여름철_오차악화_진단.md"


def stats(error2, error3, q):
    e2, e3, q = (np.asarray(a, dtype=float) for a in (error2, error3, q))
    require(e2.ndim == e3.ndim == q.ndim == 1 and len(q) > 0 and e2.shape == e3.shape == q.shape,
            "Invalid diagnostic shape")
    require(np.isfinite(np.stack([e2,e3,q])).all() and (q >= 0).all(), "Invalid diagnostic values")
    row = dict(times=len(q), price_k_sum=float(q.sum()))
    for method, e in (("B2",e2),("B3",e3)):
        row[method+"_W"] = float(np.sum(q*np.abs(e)))
        row[method+"_over_W"] = float(np.sum(q*np.maximum(e,0)))
        row[method+"_under_W"] = float(np.sum(q*np.maximum(-e,0)))
        row[method+"_weighted_bias"] = float(np.sum(q*e)/q.sum()) if q.sum() else None
        row[method+"_mae"] = float(np.abs(e).mean())
        require(np.isclose(row[method+"_W"],row[method+"_over_W"]+row[method+"_under_W"],
                           rtol=1e-12,atol=1e-5), "Direction cost mismatch")
    for key in ("W","over_W","under_W"):
        row["delta_"+key] = row["B3_"+key]-row["B2_"+key]
    require(np.isclose(row["delta_W"],row["delta_over_W"]+row["delta_under_W"],
                       rtol=1e-11,atol=1e-5), "Direction delta mismatch")
    row["change_pct"] = (row["B3_W"]/row["B2_W"]-1)*100 if row["B2_W"] else None
    active = q > 0
    row["positive_weight_times"] = int(active.sum())
    row["weighted_prediction_shift"] = float(np.sum(q*(e3-e2))/q.sum()) if q.sum() else None
    row["higher_prediction_times"] = int(np.sum(active & (e3>e2)))
    row["lower_prediction_times"] = int(np.sum(active & (e3<e2)))
    row["same_prediction_times"] = int(np.sum(active & (e3==e2)))
    row["opposite_error_times"] = int(np.sum(active & (e2*e3<0)))
    row["both_over_times"] = int(np.sum(active & (e2>0) & (e3>0)))
    row["both_under_times"] = int(np.sum(active & (e2<0) & (e3<0)))
    row["zero_error_times"] = int(np.sum(active & ((e2==0) | (e3==0))))
    require(sum(row[k] for k in ("opposite_error_times","both_over_times","both_under_times","zero_error_times"))
            == row["positive_weight_times"], "Sign partition mismatch")
    row["error_correlation"] = float(np.corrcoef(e2[active],e3[active])[0,1]) if active.sum()>1 and np.std(e2[active])>0 and np.std(e3[active])>0 else None
    return row


def concentration(daily):
    values = np.asarray(daily,dtype=float)
    require(values.ndim == 1 and len(values)>0 and np.isfinite(values).all(), "Invalid daily deltas")
    pos = np.maximum(values,0)
    result = dict(days=len(values),worse_days=int((values>0).sum()),better_days=int((values<0).sum()),
                  equal_days=int((values==0).sum()),gross_increase=float(pos.sum()),
                  gross_improvement=float(-np.minimum(values,0).sum()),net_delta=float(values.sum()))
    for n in (1,3,5):
        result[f"top_{n}_positive_days_share"] = float(np.sort(pos)[-n:].sum()/pos.sum()) if pos.sum() else None
    require(np.isclose(result["net_delta"],result["gross_increase"]-result["gross_improvement"],
                       rtol=1e-11,atol=1e-5),"Daily accounting mismatch")
    return result


def load_prediction(path, truth):
    flat=pd.read_parquet(path)
    flat["ts"]=times(flat.ts)
    unique(flat,KEY)
    require(len(flat)==253*8784 and set(flat.site_id)==set(truth.columns) and flat.ts.isin(truth.index).all(),
            "Frozen V1 grid changed")
    mat=flat.pivot(index="ts",columns="site_id",values="kwh").reindex(index=truth.index,columns=truth.columns)
    require(np.isfinite(mat.to_numpy()).all() and mat.ge(0).all().all(), "Invalid saved predictions")
    return mat


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    require(sys.version_info[:2]==(3,13) and not sysconfig.get_config_var("Py_GIL_DISABLED"), "Regular CPython 3.13 required")
    suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"),pattern="test_summer_diagnostics.py")
    test=unittest.TextTestRunner(verbosity=2).run(suite)
    require(test.wasSuccessful(),"D3 synthetic tests failed")
    require(not (OUT/"summary.json").exists(),"Completed D3 exists")
    started=now()
    prior=json.loads((B3/"summary.json").read_text(encoding="utf-8"))
    p1=json.loads((P1/"summary.json").read_text(encoding="utf-8"))
    require(prior["status"]==p1["status"]=="passed","Previous run unavailable")
    source={f:digest(SOURCE/f) for f in INPUTS}
    require(source==p1["source_sha256"],"Source drift")
    protected=dict(p1["protected_sha256"])
    for p in P1.rglob("*"):
        if p.is_file(): protected[str(p.relative_to(ROOT))]=digest(p)
    for name in ("2026-10-20_05_B3_L1손실_비교.md","2026-10-20_06_B3_최종학습과_평가절차_설계.md"):
        p=ROOT/"reports"/name
        protected[str(p.relative_to(ROOT))]=digest(p)
    require(all(digest(ROOT/rel)==h for rel,h in protected.items()),"Previous artifacts changed")
    code=[Path(__file__),ROOT/"tests/test_summer_diagnostics.py",ROOT/"scripts/validation_core.py",ROOT/"scripts/s0_validation_check.py"]
    code_hash={str(p.relative_to(ROOT)):digest(p) for p in code}
    prereg=digest(REPORT)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"preregistered_report.md").write_bytes(REPORT.read_bytes())
    (OUT/"executed_code").mkdir(exist_ok=True)
    for p in code: (OUT/"executed_code"/p.name).write_bytes(p.read_bytes())
    sites=pd.read_csv(SOURCE/"sites.csv",encoding="utf-8-sig")
    metadata(sites)
    groups=sites.set_index("site_id").current_group.map({"A":"G1","B":"G2"})
    ids=json.loads((ROOT/"data/validation/2026-10-19_B0_V1/summary.json").read_text(encoding="utf-8"))["fixed_148_sites"]
    require(len(ids)==148,"Fixed cohort drift")
    gen=pd.read_parquet(SOURCE/"train/generation.parquet",columns=KEY+["energy"])
    gen["ts"]=times(gen.ts)
    truth=truth_matrix(gen,sites.site_id,WINDOWS[0].axis())
    require(truth[ids].notna().all().all(),"Fixed cohort truth unavailable")
    price=pd.read_csv(SOURCE/"train/smp.csv",encoding="utf-8-sig")
    price["ts"]=times(price.ts);unique(price,["ts"])
    matrices={"B2":load_prediction(B2/"V1/B2/predictions_mean.parquet",truth),
              "B3":load_prediction(B3/"V1/B3/predictions_mean.parquet",truth)}
    all_axis=truth.index
    start=all_axis-pd.Timedelta(hours=1)
    # Reproduce all saved fixed-cohort annual, quarterly and monthly scores first.
    for method,mat in matrices.items():
        ref={(r["sample"],r["period"]):r for r in prior["periods"][0][method+"_scores"]}
        masks=[("all",np.ones(len(start),dtype=bool))]+[(f"Q{q}",start.quarter==q) for q in range(1,5)]+[(f"M{m:02}",start.month==m) for m in range(1,13)]
        for period,mask in masks:
            actual=local_score(truth[ids],mat[ids],groups,price,all_axis[mask])
            for key in ("times","W_local","R_local","S_local","price_k_sum"):
                require(np.isclose(actual[key],ref[("fixed_148",period)][key],rtol=1e-12,atol=1e-5),"Saved score not reproduced")
    axis=all_axis[start.quarter==3]
    target_day=(axis-pd.Timedelta(hours=1)).floor("D")
    month=(axis-pd.Timedelta(hours=1)).month
    q=price.set_index("ts").loc[axis].smp.to_numpy()*weights(axis)
    truth_total,complete=group_totals(truth.loc[axis,ids],groups)
    require(complete.all() and len(axis)==2208,"Q3 scope mismatch")
    errors={m:group_totals(mat.loc[axis,ids],groups)[0]-truth_total for m,mat in matrices.items()}
    require(int(truth.loc[axis].notna().all(axis=1).sum())==0,"Summer full cohort availability changed")
    rows=[];hour_rows=[];daily=[];paired=[]
    for group in ("G1","G2","TOTAL"):
        if group=="TOTAL":
            # Concatenate group residuals; never cancel G1 against G2.
            e2=errors["B2"][["G1","G2"]].to_numpy().ravel()
            e3=errors["B3"][["G1","G2"]].to_numpy().ravel()
            qq=np.repeat(q,2);mm=np.repeat(month,2);hh=np.repeat(axis.hour,2);kk=np.repeat(weights(axis),2)
            days=np.repeat(target_day.to_numpy(),2)
        else:
            e2=errors["B2"][group].to_numpy();e3=errors["B3"][group].to_numpy()
            qq=q;mm=month;hh=axis.hour;kk=weights(axis);days=target_day.to_numpy()
        for period,pmask in [("Q3",np.ones(len(e2),dtype=bool))]+[(f"M{m:02}",mm==m) for m in (7,8,9)]:
            for k in ("all",1,3):
                mask=pmask if k=="all" else pmask & (kk==k)
                rows.append(dict(group=group,period=period,k=str(k),**stats(e2[mask],e3[mask],qq[mask])))
        for m in (7,8,9):
            for h in range(24):
                mask=(mm==m)&(hh==h)
                hour_rows.append(dict(group=group,month=m,end_hour=h,**stats(e2[mask],e3[mask],qq[mask])))
        for day in sorted(set(days[mm==8])):
            mask=(mm==8)&(days==day)
            daily.append(dict(group=group,target_day=str(pd.Timestamp(day).date()),**stats(e2[mask],e3[mask],qq[mask])))
        if group!="TOTAL":
            for m in (7,8,9):
                for s2 in (-1,0,1):
                    for s3 in (-1,0,1):
                        mask=(mm==m)&(np.sign(e2)==s2)&(np.sign(e3)==s3)
                        if mask.any(): paired.append(dict(group=group,month=m,B2_sign=s2,B3_sign=s3,**stats(e2[mask],e3[mask],qq[mask])))
    additive=("B2_W","B3_W","delta_W","B2_over_W","B3_over_W","B2_under_W","B3_under_W","delta_over_W","delta_under_W")
    for group in ("G1","G2","TOTAL"):
        rr=[r for r in rows if r["group"]==group]
        root=next(r for r in rr if r["period"]=="Q3" and r["k"]=="all")
        for field in additive:
            require(np.isclose(sum(r[field] for r in rr if r["period"].startswith("M") and r["k"]=="all"),root[field],rtol=1e-11,atol=1e-5),"Month accounting mismatch")
        for m in (7,8,9):
            row=next(r for r in rr if r["period"]==f"M{m:02}" and r["k"]=="all")
            for field in additive:
                for parts in ([r for r in rr if r["period"]==f"M{m:02}" and r["k"]!="all"],
                              [r for r in hour_rows if r["group"]==group and r["month"]==m]):
                    require(np.isclose(sum(r[field] for r in parts),row[field],rtol=1e-11,atol=1e-5),"k/hour accounting mismatch")
                if m==8: require(np.isclose(sum(r[field] for r in daily if r["group"]==group),row[field],rtol=1e-11,atol=1e-5),"Daily accounting mismatch")
                if group!="TOTAL": require(np.isclose(sum(r[field] for r in paired if r["group"]==group and r["month"]==m),row[field],rtol=1e-11,atol=1e-5),"Sign-pair accounting mismatch")
    for row in (r for r in rows if r["group"]=="TOTAL"):
        parts=[r for r in rows if r["group"]!="TOTAL" and r["period"]==row["period"] and r["k"]==row["k"]]
        for field in additive: require(np.isclose(sum(r[field] for r in parts),row[field],rtol=1e-11,atol=1e-5),"Group accounting mismatch")
        if row["k"]=="all":
            for method in ("B2","B3"):
                ref=next(r for r in prior["periods"][0][method+"_scores"] if r["sample"]=="fixed_148" and r["period"]==row["period"])
                require(np.isclose(ref["W_local"],row[method+"_W"],rtol=1e-12),"Diagnostic W changed")
    concentration_rows=[dict(group=g,**concentration([r["delta_W"] for r in daily if r["group"]==g])) for g in ("G1","G2","TOTAL")]
    for name,values in (("periods",rows),("hours",hour_rows),("days",daily),("sign_pairs",paired)):
        pd.DataFrame(values).to_csv(OUT/f"{name}.csv",index=False,encoding="utf-8-sig")
    require(source=={f:digest(SOURCE/f) for f in INPUTS},"Source changed during diagnosis")
    require(all(digest(ROOT/rel)==h for rel,h in protected.items()),"Protected artifact changed during diagnosis")
    require(code_hash=={str(p.relative_to(ROOT)):digest(p) for p in code},"Executed code changed")
    require(prereg==digest(REPORT),"Registered scope changed during diagnosis")
    require("lightgbm" not in sys.modules,"Model library unexpectedly imported")
    write_json(OUT/"summary.json",dict(status="passed",experiment="D3_B3_summer",started_at=started,finished_at=now(),
        runtime=dict(executable=sys.executable,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
        source_sha256=source,protected_sha256=protected,code_sha256=code_hash,prereg_report_sha256=prereg,
        synthetic_tests=test.testsRun,cohort_sites=148,group_sites={g:int(groups.loc[ids].eq(g).sum()) for g in ("G1","G2")},
        Q3_times=len(axis),Q3_common253_times=0,periods=rows,hours=hour_rows,days=daily,sign_pairs=paired,
        concentration=concentration_rows,model_fit_count=0,model_prediction_count=0,weight_search_count=0,
        V2_diagnosed=False,final_evaluation_diagnosed=False))
    log("D3 passed; frozen scores reproduced; all accounting checked; no fits/inference/V2")


if __name__=="__main__":
    main()
