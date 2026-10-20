"""Preregistered R3/R4 B3 versus B5 replay. No V2/evaluation modeling or scoring."""
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
from b2_joint_compare import frames,joint_features,runtime,scores
from b3_l1_compare import parameters
from b5_loss_weighting import OUT as PREVIOUS,training_weights,load_prediction
from s0_validation_check import ROOT,SOURCE,INPUTS,baseline_contract,digest,load_forecasts,log,monthly,now,table,write_json
from validation_core import Window,KEY,times,truth_matrix,metadata,unique,require,grid,weights,local_score

OUT=ROOT/"data/validation/2026-10-20_B3_B5_rolling"
REPORT=ROOT/"reports/2026-10-20_11_B3_B5_후반기_순차검증.md"
PLAN=ROOT/"reports/성능개선_진행계획.md"
REPLAYS=(Window("R3","2024-07-01","2024-10-01","2022-07-01","2024-06-30"),
         Window("R4","2024-10-01","2025-01-01","2022-10-01","2024-09-30"))


def read_json(path):return json.loads(path.read_text(encoding="utf-8"))


def complete_ids(truth):
    require(len(truth)>0 and truth.columns.is_unique,"Invalid cohort truth grid")
    return truth.columns[truth.notna().all(axis=0)].tolist()


def decision(old,new,cohort):
    a={(r["sample"],r["period"]):r for r in old};b={(r["sample"],r["period"]):r for r in new}
    require(set(a)==set(b),"Different replay score scope")
    rows=[]
    for key,r in a.items():
        n=b[key]
        for field in ("times","first_ts","last_ts","R_local","price_k_sum"):
            require(r[field]==n[field],"Different replay score sample")
        w0=r["W_local"];w1=n["W_local"]
        rows.append(dict(sample=key[0],period=key[1],B3_W=w0,B5_W=w1,change_pct=(w1/w0-1)*100 if w0 else None))
    main=next(r for r in rows if r["sample"]==cohort and r["period"]=="all")
    months=[r for r in rows if r["sample"]==cohort and r["period"].startswith("M")]
    require(len(months)==3,"Need exactly three replay months")
    improved=main["B5_W"]<main["B3_W"] and not np.isclose(main["B5_W"],main["B3_W"],rtol=1e-12,atol=0)
    bad=[r["period"] for r in months if r["B5_W"]>1.05*r["B3_W"] and not np.isclose(r["B5_W"],1.05*r["B3_W"],rtol=1e-12,atol=0)]
    return dict(improved=bool(improved),unstable_months=bad,passes=bool(improved and not bad),rows=rows)


def source_hashes():return {p:digest(SOURCE/p) for p in INPUTS}


def protected():
    hashes=dict(read_json(PREVIOUS/"precheck.json")["protected_sha256"])
    for p in PREVIOUS.rglob("*"):
        if p.is_file():hashes[str(p.relative_to(ROOT))]=digest(p)
    p=ROOT/"reports/2026-10-20_10_B5_B6_학습가중치_비교.md"
    hashes[str(p.relative_to(ROOT))]=digest(p)
    require(all(digest(ROOT/p)==h for p,h in hashes.items()),"Previous artifact drift")
    return hashes


def code_paths():
    return [Path(__file__),ROOT/"tests/test_rolling_compare.py"]+[ROOT/"scripts"/p for p in
        ("b5_loss_weighting.py","b3_l1_compare.py","b2_joint_compare.py","b0_v1_baseline.py","s0_validation_check.py","validation_core.py")]


def inputs():
    sites=pd.read_csv(SOURCE/"sites.csv",encoding="utf-8-sig");metadata(sites)
    gen=pd.read_parquet(SOURCE/"train/generation.parquet");gen["ts"]=times(gen.ts);unique(gen,KEY)
    price=pd.read_csv(SOURCE/"train/smp.csv",encoding="utf-8-sig");price["ts"]=times(price.ts);unique(price,["ts"])
    return sites,gen,price


def availability():
    require(not (OUT/"availability.json").exists(),"Availability already recorded")
    OUT.mkdir(parents=True,exist_ok=True)
    sites,gen,price=inputs();spec=baseline_contract();rows=[]
    for w in REPLAYS:
        train=w.training(gen);truth=truth_matrix(gen,sites.site_id,w.axis());ids=complete_ids(truth)
        cohort=sites.set_index("site_id").loc[ids]
        require(set(cohort.current_group)=={"A","B"},"Replay cohort needs both groups")
        p=price.set_index("ts").reindex(w.axis()).smp
        require(np.isfinite(p).all() and p.ge(0).all(),"Replay price missing")
        d=train.merge(sites,on="site_id",how="left",validate="many_to_one",sort=False)
        elev=spec["solar"](pd.DatetimeIndex(d.ts),d.lat.to_numpy(),d.lon.to_numpy())[0]
        day=int((elev>0).sum());require(day>=10000,"Insufficient daylight training")
        rows.append(dict(window=vars(w),cohort_ids=ids,cohort_sites=len(ids),group_sites=cohort.groupby("current_group").size().to_dict(),
            group_capacity_kw=cohort.groupby("current_group").capacity_kw.sum().to_dict(),train_rows=len(train),daylight_rows=day,
            train_sites=int(train.site_id.nunique()),cohort_train_sites=int(cohort.index.isin(train.site_id.unique()).sum()),
            train_ts_min=str(train.ts.min()),train_ts_max=str(train.ts.max()),common253_times=int(truth.notna().all(axis=1).sum()),
            prediction_times=len(truth),prediction_rows=len(truth)*253,price_k_sum=float((p*weights(w.axis())).sum())))
        log(f"{w.name} availability: complete sites={len(ids)}, training sites={train.site_id.nunique()}, common253={rows[-1]['common253_times']}")
        del train,truth,d;gc.collect()
    source=source_hashes();require(source==read_json(PREVIOUS/"precheck.json")["source_sha256"],"Source drift")
    write_json(OUT/"availability.json",dict(at=now(),runtime=runtime(),source_sha256=source,periods=rows,model_fit_count=0,scores_computed=False))


def fit_predict(w,method,gen,sites,fa,fb,spec,truth,available,reference=None):
    folder=OUT/w.name/method;folder.mkdir(parents=True,exist_ok=True)
    file=folder/"fit_summary.json"
    if file.exists():
        fit=read_json(file)
        require(fit["method"]==method and fit["parameters"]==parameters() and digest(folder/"model.txt")==fit["model_sha256"],"Saved replay model drift")
    else:
        require(not (folder/"fit_journal.json").exists(),"Partial replay fit; never refit automatically")
        train=w.training(gen);base=train.merge(sites,on="site_id",how="left",validate="many_to_one",sort=False)
        d,x10,d17,x17=frames(base,fa,fb,spec);x=joint_features(d,x10,d17,x17,spec)
        day=d.elev.gt(0);x=x.loc[day];y=target_values(d.loc[day]);require(x.index.equals(y.index),"Replay target alignment")
        sample=None if method=="B3" else training_weights(d.loc[day,"cap"].to_numpy(),d.loc[day,"ts"],None,"B5")[0]
        train_info=dict(train_rows=len(train),daylight_rows=len(x),train_ts_min=str(train.ts.min()),train_ts_max=str(train.ts.max()))
        require(all(train_info[k]==available[k] for k in train_info),"Replay availability changed")
        contract=dict(**train_info,features=x.columns.tolist(),dtypes=[str(v) for v in x.dtypes],nan_rows=int(x.isna().any(axis=1).sum()))
        if reference:require(contract==reference["contract"],"B3/B5 training contract mismatch")
        write_json(folder/"fit_journal.json",dict(status="started",at=now(),model_fit_count=1))
        log(f"{w.name} {method}: fit {x.shape}");started=time.perf_counter()
        model=lgb.LGBMRegressor(**parameters()).fit(x,y,sample_weight=sample)
        require(model.booster_.num_trees()==600,"Unexpected replay tree count")
        (folder/"model.txt").write_text(model.booster_.model_to_string(),encoding="utf-8")
        fit=dict(method=method,contract=contract,parameters=parameters(),fit_seconds=time.perf_counter()-started,
            model_sha256=digest(folder/"model.txt"),sample_weight_mean=float(sample.mean()) if sample is not None else 1.,model_fit_count=1)
        write_json(file,fit);write_json(folder/"fit_journal.json",dict(status="saved",at=now(),model_fit_count=1))
        del model,train,base,d,x10,d17,x17,x,y,sample;gc.collect()
    if reference:require(fit["contract"]==reference["contract"],"Stored pair contract mismatch")
    model=lgb.Booster(model_str=(folder/"model.txt").read_text(encoding="utf-8"))
    require(model.num_trees()==600 and model.feature_name()==fit["contract"]["features"] and model.dump_model()["objective"]=="regression_l1","Stored replay model mismatch")
    path=folder/"predictions.parquet"
    if not path.exists():
        pieces=[]
        for axis in monthly(w.axis()):
            base=grid(sites,axis);d,x10,d17,x17=frames(base,fa,fb,spec);x=joint_features(d,x10,d17,x17,spec)
            require(x.columns.tolist()==fit["contract"]["features"],"Replay inference feature drift")
            pieces.append(d[KEY].assign(kwh=prediction_values(d,model.predict(x))))
            log(f"{w.name} {method}: predicted {axis[0]:%Y-%m}")
            del base,d,x10,d17,x17,x;gc.collect()
        flat=pd.concat(pieces,ignore_index=True);flat.to_parquet(path,index=False,compression="zstd")
        write_json(folder/"prediction_manifest.json",dict(sha256=digest(path),model_sha256=fit["model_sha256"]))
        del flat,pieces;gc.collect()
    pm=read_json(folder/"prediction_manifest.json")
    require(digest(path)==pm["sha256"] and pm["model_sha256"]==fit["model_sha256"],"Replay prediction drift")
    pred=load_prediction(path,truth);del model;gc.collect()
    return pred,fit


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    require(sys.version_info[:2]==(3,13) and not sysconfig.get_config_var("Py_GIL_DISABLED"),"Regular CPython 3.13 required")
    if "--availability" in sys.argv:availability();return
    tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/"tests"),pattern="test_rolling_compare.py"))
    require(tests.wasSuccessful(),"Replay contract tests failed")
    available=read_json(OUT/"availability.json")
    require(available["runtime"]==runtime() and available["source_sha256"]==source_hashes(),"Availability input drift")
    if "--precheck" in sys.argv:
        require(not (OUT/"precheck.json").exists(),"Replay precheck already frozen")
        (OUT/"executed_code").mkdir(exist_ok=True)
        for p in code_paths():(OUT/"executed_code"/p.name).write_bytes(p.read_bytes())
        (OUT/"preregistered_report.md").write_bytes(REPORT.read_bytes());(OUT/"performance_plan.md").write_bytes(PLAN.read_bytes())
        write_json(OUT/"precheck.json",dict(at=now(),availability_sha256=digest(OUT/"availability.json"),
            source_sha256=source_hashes(),protected_sha256=protected(),code_sha256={str(p.relative_to(ROOT)):digest(p) for p in code_paths()},
            report_sha256=digest(REPORT),plan_sha256=digest(PLAN),parameters=parameters(),runtime=runtime(),model_fit_count=0,fit_budget=4,scores_computed=False))
        log("P-004 precheck passed: 4-fit budget; no scores/fits");return
    require(not (OUT/"summary.json").exists(),"Replay already completed")
    pre=read_json(OUT/"precheck.json")
    def intact():
        require(pre["availability_sha256"]==digest(OUT/"availability.json") and pre["source_sha256"]==source_hashes(),"Replay input changed")
        require(pre["protected_sha256"]==protected(),"Protected result changed")
        require(pre["code_sha256"]=={str(p.relative_to(ROOT)):digest(p) for p in code_paths()},"Replay code changed")
        require(pre["report_sha256"]==digest(REPORT) and pre["plan_sha256"]==digest(PLAN),"Replay scope changed")
        require(pre["parameters"]==parameters() and pre["runtime"]==runtime(),"Replay settings changed")
    intact();started=now();sites,gen,price=inputs();groups=sites.set_index("site_id").current_group.map({"A":"G1","B":"G2"})
    spec=baseline_contract();fa,fb,excluded=load_forecasts("train",spec);outputs=[]
    # IDs are fixed from original B3 precheck, never inferred from prediction errors.
    fixed_ids=read_json(ROOT/"data/validation/2026-10-20_B2_joint/precheck.json")["V1_cohort_ids"]
    for w,av in zip(REPLAYS,available["periods"]):
        require(vars(w)==av["window"],"Replay window drift")
        truth=truth_matrix(gen,sites.site_id,w.axis());ids=av["cohort_ids"]
        require(ids==complete_ids(truth),"Replay complete cohort drift")
        result={};reference=None
        for method in ("B3","B5"):
            pred,fit=fit_predict(w,method,gen,sites,fa,fb,spec,truth,av,reference)
            if method=="B3":reference=fit
            ss=scores(truth,pred,groups,price,ids,"complete_quarter")
            diag=[]
            for name,members in (("fixed_148",[i for i in ids if i in fixed_ids]),("added_complete",[i for i in ids if i not in fixed_ids])):
                if members and set(groups.loc[members])=={"G1","G2"}:
                    diag.append(dict(sample=name,sites=len(members),**local_score(truth[members],pred[members],groups,price,w.axis())))
            for g in ("G1","G2"):
                members=[i for i in ids if groups[i]==g]
                e=(pred[members].sum(axis=1)-truth[members].sum(axis=1)).to_numpy()
                for k in ("all",1,3):
                    mask=np.ones(len(truth),dtype=bool) if k=="all" else weights(truth.index)==k
                    q=price.set_index("ts").loc[truth.index].smp.to_numpy()*weights(truth.index)
                    diag.append(dict(group=g,k=str(k),times=int(mask.sum()),W=float(np.sum(q[mask]*np.abs(e[mask]))),
                        over_W=float(np.sum(q[mask]*np.maximum(e[mask],0))),under_W=float(np.sum(q[mask]*np.maximum(-e[mask],0)))))
            total=next(r for r in ss if r["sample"]=="complete_quarter" and r["period"]=="all")["W_local"]
            require(np.isclose(sum(r["W"] for r in diag if r.get("k")=="all"),total,rtol=1e-12),"Replay group partition mismatch")
            for g in ("G1","G2"):
                d0=next(r for r in diag if r.get("group")==g and r["k"]=="all")
                for field in ("W","over_W","under_W"):
                    require(np.isclose(sum(r[field] for r in diag if r.get("group")==g and r["k"]!="all"),d0[field],rtol=1e-12),"Replay k partition mismatch")
                require(np.isclose(d0["W"],d0["over_W"]+d0["under_W"],rtol=1e-12),"Replay direction partition mismatch")
            result[method]=dict(fit=fit,scores=ss,diagnostics=diag);del pred;gc.collect()
        comparison=decision(result["B3"]["scores"],result["B5"]["scores"],"complete_quarter")
        outputs.append(dict(window=vars(w),availability=av,methods=result,comparison=comparison))
        log(f"{w.name} paired replay complete")
    intact();selected="B5" if all(p["comparison"]["passes"] for p in outputs) else "B3"
    write_json(OUT/"summary.json",dict(status="passed",started_at=started,finished_at=now(),runtime=runtime(),
        model_fit_count=4,synthetic_tests=tests.testsRun,precheck_sha256=digest(OUT/"precheck.json"),selection=selected,
        periods=outputs,nonhourly=excluded,source_unchanged=True,protected_unchanged=True,V2_executed=False,final_evaluation_executed=False))
    log(f"P-004 completed; fits=4; selected={selected}; no V2/evaluation execution")


if __name__=="__main__":main()
