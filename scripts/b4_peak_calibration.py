"""One fixed past-residual G1 peak calibration, evaluated on frozen B3 V1/T0."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import sysconfig
import unittest

import numpy as np
import pandas as pd

from b2_joint_compare import T0, fit_predict, runtime, scores, compare
from b3_l1_compare import parameters
from s0_validation_check import ROOT, SOURCE, INPUTS, baseline_contract, digest, load_forecasts, log, now, write_json
from validation_core import KST, KEY, WINDOWS, Window, clock, metadata, require, times, truth_matrix, unique, weights

OUT=ROOT/"data/validation/2026-10-20_B4_peak_calibration"
B3=ROOT/"data/validation/2026-10-20_B3_L1"
D3=ROOT/"data/validation/2026-10-20_D3_B3_summer"
REPORT=ROOT/"reports/2026-10-20_08_B4_G1피크_과거잔차_보정비교.md"
C0=Window("C0","2022-01-01","2023-01-01","2021-01-01","2021-12-31")


def weighted_median(values,q):
    values,q=(np.asarray(v,dtype=float) for v in (values,q))
    require(values.ndim==q.ndim==1 and values.shape==q.shape and len(q)>0,"Bad weighted median axes")
    require(np.isfinite(values).all() and np.isfinite(q).all() and (q>=0).all() and q.sum()>0,"Bad weighted median data")
    positive=q>0;values,q=values[positive],q[positive]
    order=np.argsort(values,kind="stable")
    return float(values[order][np.searchsorted(np.cumsum(q[order]),q.sum()/2,side="left")])


def direction(x):
    return int(np.sign(x)) if abs(x)>1e-8 else 0


def adjust(prediction,ids,capacity,b):
    require(np.isfinite(b) and len(ids)>0,"Bad calibration")
    require(prediction.columns.is_unique and prediction.index.is_unique,"Duplicate output axes")
    result=prediction.copy()
    axis=result.index[weights(result.index)==3]
    old=result.loc[axis,ids].to_numpy(dtype=float)
    cap=capacity.loc[ids].to_numpy(dtype=float)
    require(np.isfinite(old).all() and (old>=0).all() and np.isfinite(cap).all() and (cap>0).all(),"Bad calibration inputs")
    total=old.sum(axis=1);target=np.maximum(total+b*cap.sum(),0)
    new=np.empty_like(old)
    active=total>0
    new[active]=old[active]*(target[active]/total[active])[:,None]
    new[~active]=target[~active,None]*(cap/cap.sum())[None,:]
    result.loc[axis,ids]=new
    require(np.isfinite(result.to_numpy()).all() and result.ge(0).all().all(),"Invalid calibrated output")
    np.testing.assert_allclose(result.loc[axis,ids].sum(axis=1),target,rtol=1e-12,atol=1e-8)
    return result


def load_prediction(path,truth):
    flat=pd.read_parquet(path);flat["ts"]=times(flat.ts);unique(flat,KEY)
    require(len(flat)==len(truth)*253 and set(flat.site_id)==set(truth.columns) and flat.ts.isin(truth.index).all(),"Saved grid changed")
    mat=flat.pivot(index="ts",columns="site_id",values="kwh").reindex(index=truth.index,columns=truth.columns)
    require(np.isfinite(mat.to_numpy()).all() and mat.ge(0).all().all(),"Invalid saved prediction")
    return mat


def source_and_protected():
    old=json.loads((D3/"summary.json").read_text(encoding="utf-8"))
    protected=dict(old["protected_sha256"])
    for p in D3.rglob("*"):
        if p.is_file(): protected[str(p.relative_to(ROOT))]=digest(p)
    for p in (ROOT/"reports/2026-10-20_07_B3_여름철_오차악화_진단.md",):
        protected[str(p.relative_to(ROOT))]=digest(p)
    require(all(digest(ROOT/p)==h for p,h in protected.items()),"Old result drift")
    source={f:digest(SOURCE/f) for f in INPUTS}
    require(source==old["source_sha256"],"Source drift")
    return source,protected


def code_paths():
    return [Path(__file__),ROOT/"tests/test_peak_calibration.py",ROOT/"scripts/b3_l1_compare.py",
            ROOT/"scripts/b2_joint_compare.py",ROOT/"scripts/b0_v1_baseline.py",
            ROOT/"scripts/s0_validation_check.py",ROOT/"scripts/validation_core.py"]


def cohorts(gen,sites,old):
    groups=sites.set_index("site_id").current_group.map({"A":"G1","B":"G2"})
    v1ids=json.loads((ROOT/"data/validation/2026-10-19_B0_V1/summary.json").read_text(encoding="utf-8"))["fixed_148_sites"]
    t0ids=json.loads((ROOT/"data/validation/2026-10-20_B2_joint/precheck.json").read_text(encoding="utf-8"))["T0"]["cohort_ids"]
    output=[]
    price=pd.read_csv(SOURCE/"train/smp.csv",encoding="utf-8-sig");price["ts"]=times(price.ts);unique(price,["ts"])
    for cal,target,ids in ((C0,T0,t0ids),(T0,WINDOWS[0],v1ids)):
        axis=cal.axis();axis=axis[axis<=pd.Timestamp(target.cutoff,tz=KST)]
        truth=truth_matrix(gen,ids,axis)
        trained=set(cal.training(gen).site_id)
        selected=[i for i in ids if groups[i]=="G1" and i in trained and truth[i].notna().all()]
        cap=float(sites.set_index("site_id").loc[selected].capacity_kw.sum())
        q=price.set_index("ts").loc[axis].smp.to_numpy()*weights(axis)
        months=(axis-pd.Timedelta(hours=1)).quarter
        n=[int(((weights(axis)==3)&(months==quarter)&(q>0)).sum()) for quarter in (1,2,3,4)]
        eligible=len(selected)>=20 and cap>=1000 and len(axis)==8736 and min(n)>=400
        require(axis.max()<=pd.Timestamp(target.cutoff,tz=KST)<clock(target.axis()).prediction_at.min(),"Calibration target leakage")
        require(pd.Timestamp(cal.cutoff,tz=KST)<clock(axis).prediction_at.min(),"Calibration base-model leakage")
        output.append(dict(calibration_window=vars(cal),target_window=vars(target),target_ids=ids,calibration_ids=selected,
            calibration_sites=len(selected),calibration_capacity_kw=cap,times=len(axis),first_ts=str(axis.min()),last_ts=str(axis.max()),
            positive_peak_times_by_quarter=n,eligible=eligible))
    return output


def fit_calibration(truth,pred,ids,capacity,price,axis):
    require(truth.loc[axis,ids].notna().all().all(),"Missing calibration truth")
    cap=float(capacity.loc[ids].sum())
    q=price.set_index("ts").loc[axis].smp.to_numpy()*weights(axis)
    residual=(truth.loc[axis,ids].sum(axis=1)-pred.loc[axis,ids].sum(axis=1)).to_numpy()/cap
    peak=weights(axis)==3
    value=weighted_median(residual[peak],q[peak])
    quarter=(axis-pd.Timedelta(hours=1)).quarter
    parts=[weighted_median(residual[peak&(quarter==v)],q[peak&(quarter==v)]) for v in (1,2,3,4)]
    stable=direction(value)!=0 and all(direction(v)==direction(value) for v in parts)
    return dict(coefficient=value,quarter_medians=parts,stable=stable,capacity_kw=cap,sites=len(ids),
        peak_times=int(peak.sum()),positive_weight_peak_times=int((peak&(q>0)).sum()),first_ts=str(axis.min()),last_ts=str(axis.max()))


def peak_cost(truth,pred,ids,groups,price):
    axis=truth.index[((truth.index-pd.Timedelta(hours=1)).month==8)&(weights(truth.index)==3)]
    g1=[i for i in ids if groups[i]=="G1"]
    require(truth.loc[axis,g1].notna().all().all(),"Missing August truth")
    q=price.set_index("ts").loc[axis].smp.to_numpy()*weights(axis)
    error=(pred.loc[axis,g1].sum(axis=1)-truth.loc[axis,g1].sum(axis=1)).to_numpy()
    return dict(W=float(np.sum(q*np.abs(error))),over_W=float(np.sum(q*np.maximum(error,0))),
                under_W=float(np.sum(q*np.maximum(-error,0))),times=len(axis))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    require(sys.version_info[:2]==(3,13) and not sysconfig.get_config_var("Py_GIL_DISABLED"),"Regular CPython 3.13 required")
    suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"),pattern="test_peak_calibration.py")
    tests=unittest.TextTestRunner(verbosity=2).run(suite);require(tests.wasSuccessful(),"B4 contracts failed")
    old=json.loads((B3/"summary.json").read_text(encoding="utf-8"))
    require(old["status"]=="passed" and old["runtime"]==runtime() and old["parameters"]==parameters(),"Frozen B3 settings/runtime drift")
    source,protected=source_and_protected()
    if "--precheck" in sys.argv:
        require(not (OUT/"precheck.json").exists(),"Precheck already fixed")
        sites=pd.read_csv(SOURCE/"sites.csv",encoding="utf-8-sig");metadata(sites)
        gen=pd.read_parquet(SOURCE/"train/generation.parquet");gen["ts"]=times(gen.ts);unique(gen,KEY)
        scopes=cohorts(gen,sites,old)
        train=C0.training(gen);loc=sites.set_index("site_id").loc[train.site_id]
        day=baseline_contract()["solar"](times(train.ts),loc.lat.to_numpy(),loc.lon.to_numpy())[0]>0
        eligible=all(v["eligible"] for v in scopes) and int(day.sum())>=10000
        OUT.mkdir(parents=True,exist_ok=True);(OUT/"executed_code").mkdir(exist_ok=True)
        for p in code_paths(): (OUT/"executed_code"/p.name).write_bytes(p.read_bytes())
        (OUT/"preregistered_report.md").write_bytes(REPORT.read_bytes())
        write_json(OUT/"precheck.json",dict(status="eligible" if eligible else "insufficient_data",at=now(),
            scopes=scopes,C0_train_rows=len(train),C0_daylight_rows=int(day.sum()),C0_train_sites=int(train.site_id.nunique()),
            source_sha256=source,protected_sha256=protected,code_sha256={str(p.relative_to(ROOT)):digest(p) for p in code_paths()},
            prereg_report_sha256=digest(REPORT),runtime=runtime(),parameters=parameters(),model_fit_count=0,calibration_fit_count=0,
            evaluation_scores_computed=False,model_fit_budget=1))
        log(f"B4 precheck: eligible={eligible}; C0 daylight={int(day.sum()):,}; cohorts={[v['calibration_sites'] for v in scopes]}; fits/scores=0")
        return
    require(not (OUT/"summary.json").exists(),"Completed B4 exists")
    pre=json.loads((OUT/"precheck.json").read_text(encoding="utf-8"))
    require(source==pre["source_sha256"] and protected==pre["protected_sha256"] and digest(REPORT)==pre["prereg_report_sha256"],"Precheck drift")
    require(pre["code_sha256"]=={str(p.relative_to(ROOT)):digest(p) for p in code_paths()},"Code changed after precheck")
    started=now()
    if pre["status"]!="eligible":
        write_json(OUT/"summary.json",dict(status="insufficient_data",started_at=started,finished_at=now(),selection="retain_B3",
            precheck_sha256=digest(OUT/"precheck.json"),model_fit_count=0,calibration_fit_count=0,V2_executed=False))
        log("B4 ended: insufficient data; retain B3; no fits")
        return
    sites=pd.read_csv(SOURCE/"sites.csv",encoding="utf-8-sig");metadata(sites)
    cap=sites.set_index("site_id").capacity_kw
    groups=sites.set_index("site_id").current_group.map({"A":"G1","B":"G2"})
    gen=pd.read_parquet(SOURCE/"train/generation.parquet");gen["ts"]=times(gen.ts);unique(gen,KEY)
    price=pd.read_csv(SOURCE/"train/smp.csv",encoding="utf-8-sig");price["ts"]=times(price.ts);unique(price,["ts"])
    fa,fb,excluded=load_forecasts("train",baseline_contract())
    c0truth=truth_matrix(gen,sites.site_id,C0.axis())
    calpred,fits=fit_predict(C0,"B3_calibration",gen,sites,fa,fb,baseline_contract(),parameters(),c0truth,output_root=OUT)
    require(len(fits)==1 and fits[0]["daylight_rows"]==pre["C0_daylight_rows"],"Calibration fit drift")
    require(fits[0]["features"]==old["periods"][1]["fits"][0]["features"],"Calibration features drift")
    del fa,fb
    results=[];calibrations=[]
    for index,scope in enumerate(pre["scopes"]):
        window=T0 if index==0 else WINDOWS[0]
        target=truth_matrix(gen,sites.site_id,window.axis())
        base=load_prediction(B3/window.name/"B3/predictions_mean.parquet",target)
        cwindow=C0 if index==0 else T0
        ctruth=c0truth if index==0 else truth_matrix(gen,sites.site_id,T0.axis())
        cp=calpred if index==0 else load_prediction(B3/"T0/B3/predictions_mean.parquet",ctruth)
        axis=cwindow.axis();axis=axis[axis<=pd.Timestamp(window.cutoff,tz=KST)]
        calibration=fit_calibration(ctruth,cp,scope["calibration_ids"],cap,price,axis)
        calibrations.append(dict(window=window.name,**calibration))
        candidate=adjust(base,groups.index[groups.eq("G1")].tolist(),cap,calibration["coefficient"])
        pd.testing.assert_frame_equal(base.loc[:,groups.eq("G2")],candidate.loc[:,groups.eq("G2")])
        pd.testing.assert_frame_equal(base.loc[weights(base.index)==1],candidate.loc[weights(base.index)==1])
        ids=scope["target_ids"];cohort=old["periods"][1 if index==0 else 0]["cohort_name"]
        baseline=scores(target,base,groups,price,ids,cohort)
        reference={(r["sample"],r["period"]):r for r in old["periods"][1 if index==0 else 0]["B3_scores"]}
        for r in baseline:
            for k in ("times","W_local","R_local","S_local","price_k_sum"):
                require(np.isclose(r[k],reference[(r["sample"],r["period"])][k],rtol=1e-12,atol=1e-5),"Frozen B3 score drift")
        corrected=scores(target,candidate,groups,price,ids,cohort)
        comparison=compare(baseline,corrected,cohort)
        before=peak_cost(target,base,ids,groups,price);after=peak_cost(target,candidate,ids,groups,price)
        folder=OUT/window.name;folder.mkdir(exist_ok=True)
        candidate.rename_axis(index="ts",columns="site_id").stack().rename("kwh").reset_index().to_parquet(folder/"predictions_calibrated.parquet",index=False,compression="zstd")
        write_json(folder/"calibration.json",calibration)
        results.append(dict(window=vars(window),cohort_name=cohort,cohort_sites=len(ids),calibration=calibration,
            baseline_scores=baseline,corrected_scores=corrected,comparison=comparison,August_G1_peak_before=before,August_G1_peak_after=after))
        log(f"B4 {window.name}: one past coefficient applied; frozen B3 compared")
    stability=all(c["stable"] for c in calibrations) and direction(calibrations[0]["coefficient"])==direction(calibrations[1]["coefficient"])
    v1=results[1];summer_before=v1["August_G1_peak_before"]["W"];summer_after=v1["August_G1_peak_after"]["W"]
    summer=summer_after<summer_before and not np.isclose(summer_after,summer_before,rtol=1e-12,atol=0)
    q3=next(r for r in v1["comparison"]["rows"] if r["sample"]==v1["cohort_name"] and r["period"]=="Q3")
    q3_ok=q3["B2_W"]<=q3["B0_W"] or np.isclose(q3["B2_W"],q3["B0_W"],rtol=1e-12,atol=0)
    decision=dict(stable=stability,annual_and_quarters=all(p["comparison"]["passes"] for p in results),
        V1_August_G1_peak_improved=bool(summer),V1_Q3_not_worse=bool(q3_ok))
    passed=all(decision.values())
    source2,protected2=source_and_protected();require(source==source2 and protected==protected2,"Inputs changed during B4")
    require(pre["code_sha256"]=={str(p.relative_to(ROOT)):digest(p) for p in code_paths()},"Code changed during B4")
    require(digest(REPORT)==pre["prereg_report_sha256"],"Design changed during B4")
    write_json(OUT/"summary.json",dict(status="passed",started_at=started,finished_at=now(),runtime=runtime(),
        precheck_sha256=digest(OUT/"precheck.json"),model_fit_count=1,calibration_fit_count=2,synthetic_tests=tests.testsRun,
        calibration_base_fits=fits,calibrations=calibrations,periods=results,decision=decision,
        selection="B4_development_candidate" if passed else "retain_B3",V2_executed=False,final_evaluation_executed=False,
        nonhourly=excluded,source_unchanged=True,protected_unchanged=True))
    log(f"B4 completed: fits=1; selection={'B4 candidate' if passed else 'retain B3'}; no V2")


if __name__=="__main__":main()
