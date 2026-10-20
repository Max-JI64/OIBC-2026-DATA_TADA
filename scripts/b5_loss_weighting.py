"""Fixed B5 capacity and B6 capacity-price-peak L1 development ablation."""
from __future__ import annotations
import gc
import json
from pathlib import Path
import sys
import sysconfig
import time
import unittest

import lightgbm as lgb
import numpy as np
import pandas as pd

from b0_v1_baseline import target_values,prediction_values
from b2_joint_compare import T0,frames,joint_features,runtime,scores,compare
from b3_l1_compare import parameters
from s0_validation_check import ROOT,SOURCE,INPUTS,baseline_contract,digest,load_forecasts,log,monthly,now,write_json
from validation_core import KEY,WINDOWS,grid,metadata,require,times,truth_matrix,unique,weights

OUT=ROOT/"data/validation/2026-10-20_B5_B6_loss_weighting"
B3=ROOT/"data/validation/2026-10-20_B3_L1"
FINAL=ROOT/"data/final/2026-10-20_B3"
REPORT=ROOT/"reports/2026-10-20_10_B5_B6_학습가중치_비교.md"
PLAN=ROOT/"reports/성능개선_진행계획.md"


def training_weights(capacity,ts,price,method):
    cap=np.asarray(capacity,dtype=float);axis=times(ts)
    require(cap.ndim==1 and len(cap)==len(axis) and len(cap)>0 and np.isfinite(cap).all() and (cap>0).all(),"Invalid capacity weights")
    require(method in ("B5","B6"),"Unknown weighting method")
    if method=="B5":raw=cap.copy()
    else:
        unique(price,["ts"])
        p=price.set_index("ts").reindex(axis).smp.to_numpy()
        require(np.isfinite(p).all() and (p>=0).all(),"Missing or invalid historical weight price")
        raw=cap*p*weights(axis)
    mean=float(raw.mean())
    require(np.isfinite(raw).all() and mean>0 and (raw>=0).all(),"Invalid raw training weights")
    result=raw/mean
    require(np.isclose(result.mean(),1,rtol=1e-12) and np.isfinite(result).all(),"Weight normalization failed")
    return result,mean


def audit_weight(w,normalizer,raw_cf,group):
    w=np.asarray(w,dtype=float);cf=np.asarray(raw_cf,dtype=float)
    require(w.shape==cf.shape and np.isfinite(cf).all(),"Weight audit alignment failed")
    clipped=(cf<0)|(cf>1.2)
    return dict(rows=len(w),normalizer=normalizer,min=float(w.min()),median=float(np.median(w)),max=float(w.max()),
        mean=float(w.mean()),zero_rows=int((w==0).sum()),effective_rows=float(w.sum()**2/np.sum(w*w)),
        clip_low_rows=int((cf<0).sum()),clip_high_rows=int((cf>1.2).sum()),clip_weight_share=float(w[clipped].sum()/w.sum()),
        G1_weight_share=float(w[np.asarray(group)=="A"].sum()/w.sum()),G2_weight_share=float(w[np.asarray(group)=="B"].sum()/w.sum()))


def protected():
    pre=json.loads((FINAL/"precheck.json").read_text(encoding="utf-8"))
    hashes=dict(pre["protected_sha256"])
    for p in FINAL.rglob("*"):
        if p.is_file():hashes[str(p.relative_to(ROOT))]=digest(p)
    p=ROOT/"reports/2026-10-20_09_B3_최종학습과_평가예측_생성.md"
    hashes[str(p.relative_to(ROOT))]=digest(p)
    require(all(digest(ROOT/p)==h for p,h in hashes.items()),"Previous protected result changed")
    return hashes


def code_paths():
    return [Path(__file__),ROOT/"tests/test_loss_weighting.py",ROOT/"scripts/b0_v1_baseline.py",
        ROOT/"scripts/b2_joint_compare.py",ROOT/"scripts/b3_l1_compare.py",ROOT/"scripts/validation_core.py",ROOT/"scripts/s0_validation_check.py"]


def load_prediction(path,truth):
    flat=pd.read_parquet(path);flat["ts"]=times(flat.ts);unique(flat,KEY)
    require(len(flat)==len(truth)*253 and set(flat.site_id)==set(truth.columns) and flat.ts.isin(truth.index).all(),"Prediction grid mismatch")
    result=flat.pivot(index="ts",columns="site_id",values="kwh").reindex(index=truth.index,columns=truth.columns)
    require(np.isfinite(result.to_numpy()).all() and result.ge(0).all().all(),"Invalid saved predictions")
    return result


def candidate(window,method,gen,sites,fa,fb,price,spec,truth,reference_fit):
    folder=OUT/window.name/method;folder.mkdir(parents=True,exist_ok=True)
    file=folder/"fit_summary.json"
    if file.exists():
        info=json.loads(file.read_text(encoding="utf-8"))
        require(info["method"]==method and info["parameters"]==parameters() and digest(folder/"model_joint.txt")==info["model_sha256"],"Completed fit changed")
        model=lgb.Booster(model_str=(folder/"model_joint.txt").read_text(encoding="utf-8"))
    else:
        require(not (folder/"fit_journal.json").exists() and not (folder/"model_joint.txt").exists(),"Partial fit exists; never refit automatically")
        train=window.training(gen)
        base=train.merge(sites,on="site_id",how="left",validate="many_to_one",sort=False)
        d,x10,d17,x17=frames(base,fa,fb,spec);x=joint_features(d,x10,d17,x17,spec)
        day=d.elev.gt(0);x=x.loc[day];y=target_values(d.loc[day])
        require(x.index.equals(y.index),"Target/weight row alignment changed")
        require(len(base)==reference_fit["train_rows"] and len(x)==reference_fit["daylight_rows"] and x.columns.tolist()==reference_fit["features"],"Frozen sample/features changed")
        require(str(base.ts.min())==reference_fit["train_ts_min"] and str(base.ts.max())==reference_fit["train_ts_max"],"Training boundaries changed")
        w,normalizer=training_weights(d.loc[day,"cap"].to_numpy(),d.loc[day,"ts"],price,method)
        audit=audit_weight(w,normalizer,(d.loc[day,"energy"]/d.loc[day,"cap"]).to_numpy(),d.loc[day,"current_group"].to_numpy())
        nan_rows=int(x.isna().any(axis=1).sum())
        require(nan_rows==reference_fit["any_feature_nan_rows"],"NaN sample changed")
        del train,base,d,x10,d17,x17;gc.collect()
        write_json(folder/"fit_journal.json",dict(started_at=now(),model_fit_count=1,status="started"))
        log(f"{window.name} {method}: fit {x.shape}; normalized sample weights fixed")
        start=time.perf_counter();estimator=lgb.LGBMRegressor(**parameters()).fit(x,y,sample_weight=w)
        require(estimator.booster_.num_trees()==600,"Unexpected weighted tree count")
        (folder/"model_joint.txt").write_text(estimator.booster_.model_to_string(),encoding="utf-8")
        info=dict(method=method,parameters=parameters(),features=x.columns.tolist(),train_rows=reference_fit["train_rows"],daylight_rows=len(x),
            train_ts_min=reference_fit["train_ts_min"],train_ts_max=reference_fit["train_ts_max"],any_feature_nan_rows=nan_rows,
            weights=audit,fit_seconds=time.perf_counter()-start,model_fit_count=1,model_sha256=digest(folder/"model_joint.txt"))
        write_json(file,info);write_json(folder/"fit_journal.json",dict(finished_at=now(),model_fit_count=1,status="saved"))
        del estimator,x,y,w;gc.collect()
        model=lgb.Booster(model_str=(folder/"model_joint.txt").read_text(encoding="utf-8"))
    pred_path=folder/"predictions.parquet"
    if pred_path.exists():
        manifest=json.loads((folder/"prediction_manifest.json").read_text(encoding="utf-8"))
        require(digest(pred_path)==manifest["sha256"] and manifest["model_sha256"]==info["model_sha256"],"Saved predictions changed")
        return load_prediction(pred_path,truth),info
    pieces=[]
    for axis in monthly(window.axis()):
        base=grid(sites,axis);d,x10,d17,x17=frames(base,fa,fb,spec);x=joint_features(d,x10,d17,x17,spec)
        require(x.columns.tolist()==info["features"],"Inference feature order changed")
        pieces.append(d[KEY].assign(kwh=prediction_values(d,model.predict(x))))
        del base,d,x10,d17,x17,x
        log(f"{window.name} {method}: predicted {axis[0]:%Y-%m}")
    flat=pd.concat(pieces,ignore_index=True);unique(flat,KEY)
    require(len(flat)==len(truth)*253,"Incomplete weighted prediction grid")
    flat.to_parquet(pred_path,index=False,compression="zstd")
    write_json(folder/"prediction_manifest.json",dict(sha256=digest(pred_path),model_sha256=info["model_sha256"]))
    result=load_prediction(pred_path,truth)
    del model,pieces,flat;gc.collect()
    return result,info


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    require(sys.version_info[:2]==(3,13) and not sysconfig.get_config_var("Py_GIL_DISABLED"),"Regular CPython 3.13 required")
    suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"),pattern="test_loss_weighting.py")
    tests=unittest.TextTestRunner(verbosity=2).run(suite);require(tests.wasSuccessful(),"Weight contracts failed")
    old=json.loads((B3/"summary.json").read_text(encoding="utf-8"))
    require(old["status"]=="passed" and old["runtime"]==runtime() and old["parameters"]==parameters(),"Reference settings/runtime drift")
    source={f:digest(SOURCE/f) for f in INPUTS}
    require(source==json.loads((FINAL/"precheck.json").read_text(encoding="utf-8"))["source_sha256"],"Source drift")
    frozen=protected()
    if "--precheck" in sys.argv:
        require(not (OUT/"precheck.json").exists(),"Precheck already frozen")
        OUT.mkdir(parents=True,exist_ok=True);(OUT/"executed_code").mkdir(exist_ok=True)
        for p in code_paths():(OUT/"executed_code"/p.name).write_bytes(p.read_bytes())
        (OUT/"preregistered_report.md").write_bytes(REPORT.read_bytes());(OUT/"performance_plan.md").write_bytes(PLAN.read_bytes())
        write_json(OUT/"precheck.json",dict(at=now(),source_sha256=source,protected_sha256=frozen,
            code_sha256={str(p.relative_to(ROOT)):digest(p) for p in code_paths()},prereg_report_sha256=digest(REPORT),
            performance_plan_sha256=digest(PLAN),parameters=parameters(),runtime=runtime(),model_fit_count=0,fit_budget=4,scores_computed=False))
        log("P-001 precheck passed; B5/B6 two-period budget=4; fits/scores=0")
        return
    require(not (OUT/"summary.json").exists(),"Completed weighting experiment exists")
    pre=json.loads((OUT/"precheck.json").read_text(encoding="utf-8"))
    require(pre["source_sha256"]==source and pre["protected_sha256"]==frozen and pre["prereg_report_sha256"]==digest(REPORT),"Preregistration changed")
    require(pre["code_sha256"]=={str(p.relative_to(ROOT)):digest(p) for p in code_paths()},"Code changed after precheck")
    require(pre["performance_plan_sha256"]==digest(PLAN),"Performance plan changed before execution")
    started=now();spec=baseline_contract()
    sites=pd.read_csv(SOURCE/"sites.csv",encoding="utf-8-sig");metadata(sites)
    gen=pd.read_parquet(SOURCE/"train/generation.parquet");gen["ts"]=times(gen.ts);unique(gen,KEY)
    price=pd.read_csv(SOURCE/"train/smp.csv",encoding="utf-8-sig");price["ts"]=times(price.ts);unique(price,["ts"])
    require(np.isfinite(price.smp).all() and price.smp.ge(0).all(),"Invalid historical weight price")
    groups=sites.set_index("site_id").current_group.map({"A":"G1","B":"G2"})
    oldpre=json.loads((ROOT/"data/validation/2026-10-20_B2_joint/precheck.json").read_text(encoding="utf-8"))
    fa,fb,excluded=load_forecasts("train",spec)
    outputs=[]
    for index,window in enumerate((WINDOWS[0],T0)):
        truth=truth_matrix(gen,sites.site_id,window.axis())
        ids=oldpre["V1_cohort_ids"] if index==0 else oldpre["T0"]["cohort_ids"]
        require(truth[ids].notna().all().all(),"Fixed cohort truth unavailable")
        cohort=old["periods"][index]["cohort_name"]
        baseline=load_prediction(B3/window.name/"B3/predictions_mean.parquet",truth)
        baseline_scores=scores(truth,baseline,groups,price,ids,cohort)
        ref={(r["sample"],r["period"]):r for r in old["periods"][index]["B3_scores"]}
        for r in baseline_scores:
            for key in ("W_local","R_local","S_local","price_k_sum","times"):
                require(np.isclose(r[key],ref[(r["sample"],r["period"])][key],rtol=1e-12,atol=1e-5),"Frozen B3 scores changed")
        candidates={}
        for method in ("B5","B6"):
            prediction,fit=candidate(window,method,gen,sites,fa,fb,price,spec,truth,old["periods"][index]["fits"][0])
            new=scores(truth,prediction,groups,price,ids,cohort)
            candidates[method]=dict(fit=fit,scores=new,comparison=compare(baseline_scores,new,cohort))
            diag=[]
            axis=truth.index;start=axis-pd.Timedelta(hours=1)
            for month in ("all",8):
                for g in ("G1","G2"):
                    for k in ("all",1,3):
                        mask=np.ones(len(axis),dtype=bool) if month=="all" else start.month==month
                        if k!="all":mask=mask&(weights(axis)==k)
                        selected=axis[mask];members=[i for i in ids if groups[i]==g]
                        q=price.set_index("ts").loc[selected].smp.to_numpy()*weights(selected)
                        err=(prediction.loc[selected,members].sum(axis=1)-truth.loc[selected,members].sum(axis=1)).to_numpy()
                        olderr=(baseline.loc[selected,members].sum(axis=1)-truth.loc[selected,members].sum(axis=1)).to_numpy()
                        diag.append(dict(month=str(month),group=g,k=str(k),times=len(selected),B3_W=float(np.sum(q*np.abs(olderr))),
                            W=float(np.sum(q*np.abs(err))),over_W=float(np.sum(q*np.maximum(err,0))),under_W=float(np.sum(q*np.maximum(-err,0)))))
            candidates[method]["diagnostics"]=diag
            for month in ("all",8):
                period="all" if month=="all" else "M08"
                for k in ("W","B3_W"):
                    total=sum(r[k] for r in diag if r["month"]==str(month) and r["k"]=="all")
                    ref_scores=new if k=="W" else baseline_scores
                    ref_total=next(r["W_local"] for r in ref_scores if r["sample"]==cohort and r["period"]==period)
                    require(np.isclose(total,ref_total,rtol=1e-12,atol=1e-5),"Diagnostic group total mismatch")
                for g in ("G1","G2"):
                    d0=next(r for r in diag if r["month"]==str(month) and r["group"]==g and r["k"]=="all")
                    for key in ("W","B3_W","over_W","under_W"):
                        require(np.isclose(sum(r[key] for r in diag if r["month"]==str(month) and r["group"]==g and r["k"]!="all"),d0[key],rtol=1e-12,atol=1e-5),"Diagnostic k partition mismatch")
                    require(np.isclose(d0["W"],d0["over_W"]+d0["under_W"],rtol=1e-12,atol=1e-5),"Diagnostic direction sum mismatch")
            log(f"{window.name} {method}: fixed comparison completed")
        outputs.append(dict(window=vars(window),cohort_name=cohort,cohort_sites=len(ids),baseline_scores=baseline_scores,candidates=candidates))
    ranking=[]
    for method in ("B5","B6"):
        passes=all(p["candidates"][method]["comparison"]["passes"] for p in outputs)
        ratios=[]
        for p in outputs:
            r=next(r for r in p["candidates"][method]["comparison"]["rows"] if r["sample"]==p["cohort_name"] and r["period"]=="all")
            ratios.append(r["B2_W"]/r["B0_W"])
        ranking.append(dict(method=method,passes=passes,geometric_ratio=float(np.sqrt(np.prod(ratios)))))
    selected="B3";best=1.
    for r in ranking:
        if r["passes"] and r["geometric_ratio"]<best and not np.isclose(r["geometric_ratio"],best,rtol=1e-12,atol=0):selected=r["method"];best=r["geometric_ratio"]
    require(source=={f:digest(SOURCE/f) for f in INPUTS} and frozen==protected(),"Input/result drift during comparison")
    require(pre["code_sha256"]=={str(p.relative_to(ROOT)):digest(p) for p in code_paths()},"Code changed during run")
    require(pre["prereg_report_sha256"]==digest(REPORT),"Scope changed during run")
    require(pre["performance_plan_sha256"]==digest(PLAN),"Performance plan changed during run")
    write_json(OUT/"summary.json",dict(status="passed",started_at=started,finished_at=now(),runtime=runtime(),precheck_sha256=digest(OUT/"precheck.json"),
        synthetic_tests=tests.testsRun,model_fit_count=4,ranking=ranking,selection=selected,periods=outputs,
        V2_executed=False,final_evaluation_executed=False,source_unchanged=True,protected_unchanged=True,nonhourly=excluded))
    log(f"P-001 completed: 4 fits; selected={selected}; no V2/final execution")


if __name__=="__main__":main()
