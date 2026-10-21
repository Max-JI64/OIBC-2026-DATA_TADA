"""P-005/H1: one deterministic feature-axis change, four preregistered fits."""
from __future__ import annotations

import gc
import json
from pathlib import Path
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
from forecast_residual_eda import OUT as EDA, WINDOWS as EDA_WINDOWS, cohort_ids, inventory, pred_path, read
from s0_validation_check import ROOT, SOURCE, baseline_contract, digest, load_forecasts, log, monthly, now, write_json
from validation_core import KEY, clock, grid, metadata, prepare_forecast, require, times, truth_matrix, unique, weights

OUT=ROOT/"data/validation/2026-10-21_H1_cloud_interaction"
REPORT=ROOT/"reports/2026-10-21_04_H1_운량_태양계절_교차특징_비교.md"
PLAN=ROOT/"reports/성능개선_진행계획.md"
HANDOFF=ROOT/"reports/EDA_개선가설_모델링_인계.md"
WINDOWS=tuple(next(w for w in EDA_WINDOWS if w.name==name) for name in ("T0","V1","R3","R4"))
NEW=["cloud_solar","cloud_solar_sin","cloud_solar_cos"]


def h1_features(x):
    require(x.shape[1]==39 and not any(c in x for c in NEW),"H1 needs the unchanged B3 39-column frame")
    require(np.isfinite(x[["cosz","doy_sin","doy_cos"]]).all().all(),"Invalid solar/calendar inputs")
    a=x.wa_cloud_r17.astype("float64"); b=x.wb_cloud_r17.astype("float64")
    valid=np.isfinite(a)&np.isfinite(b)&a.between(0,100)&b.between(0,100)
    m=((a+b)/200).where(valid)
    z=x.cosz.astype("float64").clip(lower=0)
    y=x.copy()
    y[NEW[0]]=(m*z).astype("float32")
    y[NEW[1]]=(m*z*x.doy_sin.astype("float64")).astype("float32")
    y[NEW[2]]=(m*z*x.doy_cos.astype("float64")).astype("float32")
    pd.testing.assert_frame_equal(y.iloc[:,:39],x)
    require(y.columns[-3:].tolist()==NEW and y.dtypes.eq("float32").all(),"H1 feature order/dtypes changed")
    require(not np.isinf(y[NEW].to_numpy()).any(),"Infinite additional feature value")
    require(y[NEW].isna().eq(~valid,axis=0).all().all(),"H1 missing mask changed")
    return y


def gate(old,new,name,annual):
    a={(r["sample"],r["period"]):r for r in old};b={(r["sample"],r["period"]):r for r in new}
    require(set(a)==set(b),"Score sample sets differ")
    rows=[]
    for key,r in a.items():
        n=b[key]
        for field in ("times","first_ts","last_ts","R_local","price_k_sum"):
            require(r[field]==n[field],f"Different score scope: {field}")
        w0,w1=r["W_local"],n["W_local"]
        rows.append(dict(sample=key[0],period=key[1],times=r["times"],B3_W=w0,H1_W=w1,
                         change_pct=(w1/w0-1)*100 if w0 else None))
    total=next(r for r in rows if r["sample"]==name and r["period"]=="all")
    prefix="Q" if annual else "M"
    stability=[r for r in rows if r["sample"]==name and r["period"].startswith(prefix)]
    require(len(stability)==(4 if annual else 3),"Incomplete stability periods")
    improved=total["H1_W"]<total["B3_W"] and not np.isclose(total["H1_W"],total["B3_W"],rtol=1e-12,atol=0)
    bad=[r["period"] for r in stability if r["H1_W"]>r["B3_W"]*1.05 and
         not np.isclose(r["H1_W"],r["B3_W"]*1.05,rtol=1e-12,atol=0)]
    return dict(improved=bool(improved),unstable=bad,passes=bool(improved and not bad),rows=rows)


def protected():
    value=inventory()
    require(value==read(EDA/"precheck.json")["protected_sha256"],"Historical input drift since EDA")
    require("**모델링 가능**" in HANDOFF.read_text(encoding="utf-8"),"Completed modeling handoff required")
    es=read(EDA/"summary.json");ev=read(EDA/"verification.json");tv=read(EDA/"table_verification.json")
    require(es["status"]==ev["status"]==tv["status"]=="passed","EDA not verified")
    for name,h in ev["output_sha256"].items():
        require(digest(EDA/name)==h,"EDA aggregate drift")
    require(digest(ROOT/"scripts/forecast_residual_eda.py")==es["source_code_sha256"],"EDA code drift")
    rm=read(EDA/"report_manifest.json")
    require(digest(ROOT/"reports/2026-10-21_03_예보조건별_잔차_EDA.md")==rm["report_sha256"],"EDA report drift")
    for p in EDA.rglob("*"):
        if p.is_file():value[str(p.relative_to(ROOT))]=digest(p)
    for p in (HANDOFF,ROOT/"reports/2026-10-21_03_예보조건별_잔차_EDA.md",ROOT/"scripts/verify_forecast_residual_eda.py"):
        value[str(p.relative_to(ROOT))]=digest(p)
    return value


def code_paths():
    return [Path(__file__),ROOT/"tests/test_h1_cloud_interaction.py"]+[ROOT/"scripts"/name for name in
        ("b0_v1_baseline.py","b2_joint_compare.py","b3_l1_compare.py","s0_validation_check.py","validation_core.py","forecast_residual_eda.py")]


def inputs():
    sites=pd.read_csv(SOURCE/"sites.csv",encoding="utf-8-sig");metadata(sites)
    # Never use 2025 V2 truth or supplied evaluation forecasts for this experiment.
    gen=pd.read_parquet(SOURCE/"train/generation.parquet",filters=[("ts","<=",pd.Timestamp("2025-01-01",tz="Asia/Seoul"))])
    gen["ts"]=times(gen.ts);unique(gen,KEY)
    price=pd.read_csv(SOURCE/"train/smp.csv",encoding="utf-8-sig");price["ts"]=times(price.ts)
    price=price.loc[price.ts.le(pd.Timestamp("2025-01-01",tz="Asia/Seoul"))].copy();unique(price,["ts"])
    return sites,gen,price


def development_forecasts(spec):
    results=[];excluded={}
    for p,label in (("wa","WA_COLS"),("wb","WB_COLS")):
        station=p+"_station"
        raw=pd.read_parquet(SOURCE/f"train/forecast_{p}.parquet",columns=[station,"round","ts"]+spec[label],
                            filters=[("ts","<=",pd.Timestamp("2025-01-01",tz="Asia/Seoul"))])
        clean,n=prepare_forecast(raw,station,spec[label]);results.append(clean);excluded[p]=n
    return *results,excluded


def reference(w):
    p=pred_path(w.name,"B3"); folder=p.parent
    raw=read(folder/"fit_summary.json")
    if isinstance(raw,list):
        require(len(raw)==1 and raw[0]["mode"]=="joint","Annual B3 fit ambiguous")
        f=raw[0]
        c={k:f[k] for k in ("train_rows","daylight_rows","train_ts_min","train_ts_max","features")}
        c["nan_rows"]=f["any_feature_nan_rows"]
        c["dtypes"]=["float32"]*39
        model=folder/"model_joint.txt"
        summary=read(folder.parents[1]/"summary.json")
        require(summary["parameters"]==parameters() and summary["runtime"]==runtime(),"Annual B3 settings/runtime mismatch")
        period=next(a for a in summary["periods"] if a["window"]["name"]==w.name)
    else:
        f=raw;c=f["contract"];model=folder/"model.txt"
        require(f["parameters"]==parameters(),"Rolling B3 settings mismatch")
        summary=read(folder.parents[1]/"summary.json")
        require(summary["runtime"]==runtime(),"Rolling B3 runtime mismatch")
        period=next(a for a in summary["periods"] if a["window"]["name"]==w.name)
    require(period["window"]==vars(w),"B3 training window differs")
    require(digest(model)==f["model_sha256"],"B3 model bytes differ")
    booster=lgb.Booster(model_str=model.read_text(encoding="utf-8"))
    require(booster.num_trees()==600 and booster.feature_name()==c["features"] and booster.dump_model()["objective"]=="regression_l1","B3 serialized contract mismatch")
    del booster
    expected=period["B3_scores"] if w.name in ("T0","V1") else period["methods"]["B3"]["scores"]
    return dict(contract=c,model_sha256=digest(model),prediction_sha256=digest(p),scores=expected)


def training(w,gen,sites,fa,fb,spec,ref):
    t=w.training(gen);base=t.merge(sites,on="site_id",how="left",validate="many_to_one",sort=False)
    d,x10,d17,x17=frames(base,fa,fb,spec);x=joint_features(d,x10,d17,x17,spec)
    day=d.elev.gt(0); orig=x.loc[day]
    c=dict(train_rows=len(t),daylight_rows=int(day.sum()),train_ts_min=str(t.ts.min()),train_ts_max=str(t.ts.max()),
           features=orig.columns.tolist(),dtypes=list(map(str,orig.dtypes)),nan_rows=int(orig.isna().any(axis=1).sum()))
    require(c==ref["contract"],"H1 training differs from B3 before adding features")
    y=target_values(d.loc[day]);xx=h1_features(orig)
    require(xx.index.equals(y.index),"Training target alignment changed")
    audit=dict(**c,new_features=xx.columns.tolist(),new_dtypes=list(map(str,xx.dtypes)),
               new_nan_rows=int(xx[NEW].isna().any(axis=1).sum()),all_nan_rows=int(xx.isna().any(axis=1).sum()),
               first_prediction_at=str(clock(w.axis()).prediction_at.min()),
               raw_target_sha256=__import__("hashlib").sha256(y.to_numpy().tobytes()).hexdigest())
    del t,base,d,x10,d17,x17,x,orig;gc.collect()
    return xx,y,audit


def precheck():
    OUT.mkdir(parents=True,exist_ok=True)
    require(not (OUT/"precheck.json").exists(),"Preregistration already exists")
    hashes=protected();sites,gen,price=inputs();spec=baseline_contract();fa,fb,n=development_forecasts(spec)
    rows=[]
    for w in WINDOWS:
        ref=reference(w);x,y,audit=training(w,gen,sites,fa,fb,spec,ref)
        truth=truth_matrix(gen,sites.site_id,w.axis());ids=cohort_ids(w.name)
        require(truth[ids].notna().all().all(),"Fixed cohort missing truth")
        rows.append(dict(window=vars(w),cohort_ids=ids,cohort_name="fixed_cohort",cohort_sites=len(ids),
                         training=audit,reference=ref,common253_times=int(truth.notna().all(axis=1).sum())))
        log(f"H1 precheck {w.name}: same B3 rows, features=39+3, no fits")
        del x,y,truth;gc.collect()
    hash_lines=["", "## 실행 전 해시 고정", "", "아래 SHA-256은 사전 점검에서 계산한 값이다. 모델 적합은 아직 0회이며 전체 보존 목록과 4구간 학습 계약은 로컬 precheck.json에 고정한다.", "",
                "| 파일 | SHA-256 |", "|---|---|"]
    hash_paths=[SOURCE/p for p in ("sites.csv","train/generation.parquet","train/forecast_wa.parquet","train/forecast_wb.parquet","train/smp.csv","baseline.py")]+code_paths()
    hash_lines += [f"| `{p.relative_to(ROOT)}` | `{digest(p)}` |" for p in hash_paths]
    REPORT.write_text(REPORT.read_text(encoding="utf-8")+"\n".join(hash_lines)+"\n",encoding="utf-8")
    (OUT/"executed_code").mkdir(exist_ok=True)
    for p in code_paths():(OUT/"executed_code"/p.name).write_bytes(p.read_bytes())
    (OUT/"preregistered_report.md").write_bytes(REPORT.read_bytes());(OUT/"performance_plan.md").write_bytes(PLAN.read_bytes())
    require(protected()==hashes,"Inputs changed in precheck")
    write_json(OUT/"precheck.json",dict(at=now(),experiment="P-005_H1",runtime=runtime(),parameters=parameters(),periods=rows,
                                       protected_sha256=hashes,code_sha256={str(p.relative_to(ROOT)):digest(p) for p in code_paths()},
                                       report_sha256=digest(REPORT),plan_sha256=digest(PLAN),fit_budget=4,model_fit_count=0,
                                       feature_statistics_fit_count=0,reference_refit_count=0,nonhourly_train=n))
    log("H1 precheck passed: 4-fit budget, B3 reusable, no new fit/score")


def intact(pre,scope=True):
    require(pre["runtime"]==runtime() and pre["parameters"]==parameters(),"H1 runtime/settings changed")
    require(pre["protected_sha256"]==protected(),"Protected artifacts changed")
    for rel,h in pre["code_sha256"].items():
        require(digest(ROOT/rel)==h and digest(OUT/"executed_code"/Path(rel).name)==h,"Frozen H1 code changed")
    require(digest(OUT/"preregistered_report.md")==pre["report_sha256"] and digest(OUT/"performance_plan.md")==pre["plan_sha256"],"Frozen scope changed")
    if scope:
        require(digest(REPORT)==pre["report_sha256"] and digest(PLAN)==pre["plan_sha256"],"Live scope changed before experiment completed")


def load_prediction(path,truth):
    f=pd.read_parquet(path);f["ts"]=times(f.ts);unique(f,KEY)
    require(len(f)==len(truth)*253 and set(f.site_id)==set(truth.columns) and set(f.ts)==set(truth.index),"Stored prediction grid mismatch")
    require(np.isfinite(f.kwh).all() and f.kwh.ge(0).all(),"Stored prediction nonfinite/negative")
    return f.pivot(index="ts",columns="site_id",values="kwh").reindex(index=truth.index,columns=truth.columns)


def fit_predict(w,av,gen,sites,fa,fb,spec,truth):
    folder=OUT/w.name;folder.mkdir(parents=True,exist_ok=True)
    infofile=folder/"fit_summary.json";modelpath=folder/"model.txt";journal=folder/"fit_journal.json"
    if infofile.exists():
        info=read(infofile)
        require(info["training"]==av["training"] and info["parameters"]==parameters() and digest(modelpath)==info["model_sha256"],"Completed H1 fit changed")
        log(f"{w.name}: reuse completed candidate model; never refit")
    else:
        require(not journal.exists() and not modelpath.exists(),"Partial fit exists: stop; never automatically retry fitting")
        x,y,audit=training(w,gen,sites,fa,fb,spec,av["reference"])
        require(audit==av["training"],"Frozen training inputs changed")
        write_json(journal,dict(status="started",started_at=now(),fit_attempts=1))
        start=time.perf_counter();log(f"{w.name}: H1 fit starts, shape={x.shape}, budget 1")
        try:
            model=lgb.LGBMRegressor(**parameters()).fit(x,y)
            require(model.booster_.num_trees()==600,"Unexpected H1 tree count")
            modelpath.write_text(model.booster_.model_to_string(),encoding="utf-8")
            info=dict(method="H1",training=audit,parameters=parameters(),model_sha256=digest(modelpath),
                      fit_seconds=time.perf_counter()-start,fit_attempts=1,model_fit_count=1)
            write_json(infofile,info);write_json(journal,dict(status="saved",finished_at=now(),fit_attempts=1))
            del model,x,y;gc.collect()
        except BaseException:
            write_json(journal,dict(status="failed",finished_at=now(),fit_attempts=1,error=traceback.format_exc()))
            raise
    model=lgb.Booster(model_str=modelpath.read_text(encoding="utf-8"))
    require(model.num_trees()==600 and model.feature_name()==av["training"]["new_features"] and model.dump_model()["objective"]=="regression_l1","Saved H1 model contract differs")
    path=folder/"predictions.parquet"
    if not path.exists():
        parts=[]
        for axis in monthly(w.axis()):
            d,x10,d17,x17=frames(grid(sites,axis),fa,fb,spec)
            x=h1_features(joint_features(d,x10,d17,x17,spec))
            require(x.columns.tolist()==av["training"]["new_features"],"Prediction feature order differs")
            parts.append(d[KEY].assign(kwh=prediction_values(d,model.predict(x))))
            log(f"{w.name}: H1 inferred {axis[0]:%Y-%m}")
            del d,x10,d17,x17,x;gc.collect()
        pd.concat(parts,ignore_index=True).to_parquet(path,index=False,compression="zstd")
        write_json(folder/"prediction_manifest.json",dict(sha256=digest(path),model_sha256=info["model_sha256"]))
        del parts;gc.collect()
    pm=read(folder/"prediction_manifest.json")
    require(pm==dict(sha256=digest(path),model_sha256=info["model_sha256"]),"Candidate prediction hash differs")
    pred=load_prediction(path,truth);del model;gc.collect()
    return pred,info


def diagnostics(truth,pred,sites,price,ids):
    rows=[];groups=sites.set_index("site_id").current_group.map({"A":"G1","B":"G2"})
    fixed=set(cohort_ids("V1"))
    for cohort,members in (("complete",ids),("fixed_148",[i for i in ids if i in fixed]),
                           ("added_complete",[i for i in ids if i not in fixed])):
        if not members:continue
        q=price.set_index("ts").smp.reindex(truth.index).to_numpy()*weights(truth.index)
        for g in ("G1","G2"):
            gi=[i for i in members if groups[i]==g]
            require(len(gi)>0,"Empty diagnostic group")
            error=pred[gi].sum(axis=1)-truth[gi].sum(axis=1)
            for k in ("all",1,3):
                mask=np.ones(len(q),bool) if k=="all" else weights(truth.index)==k
                e=error.to_numpy()[mask];qq=q[mask]
                rows.append(dict(cohort=cohort,group=g,k=str(k),sites=len(gi),times=int(mask.sum()),
                                 W=float(np.sum(qq*np.abs(e))),over_W=float(np.sum(qq*np.maximum(e,0))),
                                 under_W=float(np.sum(qq*np.maximum(-e,0)))))
            pieces=[r for r in rows if r["cohort"]==cohort and r["group"]==g]
            total=next(r for r in pieces if r["k"]=="all")
            for field in ("W","over_W","under_W"):
                require(np.isclose(sum(r[field] for r in pieces if r["k"]!="all"),total[field],rtol=1e-12),"Diagnostic k partition changed")
            require(np.isclose(total["W"],total["over_W"]+total["under_W"],rtol=1e-12),"Diagnostic direction split changed")
    return rows


def execute():
    require(not (OUT/"summary.json").exists(),"Completed experiment exists; use --verify")
    pre=read(OUT/"precheck.json");intact(pre)
    started=now();sites,gen,price=inputs();spec=baseline_contract();fa,fb,n=development_forecasts(spec)
    groups=sites.set_index("site_id").current_group.map({"A":"G1","B":"G2"})
    outputs=[]
    for w,av in zip(WINDOWS,pre["periods"]):
        require(vars(w)==av["window"],"Execution window differs")
        truth=truth_matrix(gen,sites.site_id,w.axis());ids=av["cohort_ids"]
        require(truth[ids].notna().all().all(),"Incomplete fixed truth")
        old=load_prediction(pred_path(w.name,"B3"),truth)
        new,fit=fit_predict(w,av,gen,sites,fa,fb,spec,truth)
        before=scores(truth,old,groups,price,ids,"fixed_cohort")
        after=scores(truth,new,groups,price,ids,"fixed_cohort")
        historical={(r["period"],r["sample"]=="common_253"):r for r in av["reference"]["scores"]}
        for r in before:
            original=historical[(r["period"],r["sample"]=="common_253")]
            for key in ("W_local","R_local","S_local","price_k_sum","times"):
                require(np.isclose(r[key],original[key],rtol=1e-12,atol=1e-5),"B3 score reproduction failed")
        comparison=gate(before,after,"fixed_cohort",w.name in ("T0","V1"))
        diag={"B3":diagnostics(truth,old,sites,price,ids),"H1":diagnostics(truth,new,sites,price,ids)}
        for method,ss in (("B3",before),("H1",after)):
            group_w=sum(r["W"] for r in diag[method] if r["cohort"]=="complete" and r["k"]=="all")
            total=next(r["W_local"] for r in ss if r["sample"]=="fixed_cohort" and r["period"]=="all")
            require(np.isclose(group_w,total,rtol=1e-12),"G1/G2 W does not sum")
        output=dict(window=vars(w),sites=len(ids),fit=fit,B3_scores=before,H1_scores=after,comparison=comparison,diagnostics=diag)
        write_json(OUT/w.name/"result.json",output);outputs.append(output)
        log(f"{w.name}: saved model/prediction/scores; no settings changed")
        del truth,old,new;gc.collect()
    intact(pre)
    require(sum(r["fit"]["model_fit_count"] for r in outputs)==4,"Fit budget differs")
    write_json(OUT/"summary.json",dict(status="passed",experiment="P-005_H1",started_at=started,finished_at=now(),
                                       runtime=runtime(),model_fit_count=4,fit_attempts=4,fit_failures=0,reference_refit_count=0,
                                       selection="H1" if all(p["comparison"]["passes"] for p in outputs) else "B3",
                                       periods=outputs,protected_unchanged=True,V2_executed=False,evaluation_executed=False,
                                       precheck_sha256=digest(OUT/"precheck.json")))
    log("H1 experiment complete: 4 fits; selection saved; V2/evaluation untouched")


def verify():
    pre=read(OUT/"precheck.json");s=read(OUT/"summary.json");intact(pre,scope=False)
    require(s["status"]=="passed" and s["model_fit_count"]==s["fit_attempts"]==4 and s["fit_failures"]==0,"Completion/fit count differs")
    sites,gen,price=inputs();groups=sites.set_index("site_id").current_group.map({"A":"G1","B":"G2"})
    for w,av,value in zip(WINDOWS,pre["periods"],s["periods"]):
        truth=truth_matrix(gen,sites.site_id,w.axis());ids=av["cohort_ids"]
        fit=read(OUT/w.name/"fit_summary.json")
        require(fit==value["fit"] and digest(OUT/w.name/"model.txt")==fit["model_sha256"],"Saved fit changed")
        booster=lgb.Booster(model_str=(OUT/w.name/"model.txt").read_text(encoding="utf-8"))
        require(booster.num_trees()==600 and booster.feature_name()==av["training"]["new_features"] and booster.dump_model()["objective"]=="regression_l1","Saved model schema differs")
        p=OUT/w.name/"predictions.parquet";pm=read(p.parent/"prediction_manifest.json")
        require(digest(p)==pm["sha256"] and pm["model_sha256"]==fit["model_sha256"],"Saved prediction changed")
        res={}
        for method,path in (("B3",pred_path(w.name,"B3")),("H1",p)):
            pred=load_prediction(path,truth)
            ss=scores(truth,pred,groups,price,ids,"fixed_cohort")
            require(ss==value[method+"_scores"],"Saved scores differ on reread")
            require(diagnostics(truth,pred,sites,price,ids)==value["diagnostics"][method],"Saved group/k directions differ")
            res[method]=ss
        require(gate(res["B3"],res["H1"],"fixed_cohort",w.name in ("T0","V1"))==value["comparison"],"Saved decision differs")
        del booster,truth,pred;gc.collect()
    require(s["selection"]==("H1" if all(p["comparison"]["passes"] for p in s["periods"]) else "B3"),"Overall choice differs")
    write_json(OUT/"verification.json",dict(at=now(),status="passed",models=4,predictions=4,new_fits=0,
                                           keys_grids_finite_nonnegative=True,score_and_decision_reproduced=True,
                                           monthly_quarter_k_group_direction_sums=True,protected_files=len(pre["protected_sha256"])))
    print(json.dumps(dict(status="passed",selection=s["selection"],model_fit_count=4,
                          periods=[dict(window=p["window"]["name"],**{k:p["comparison"][k] for k in ("passes","improved","unstable")},
                                        total=next(r for r in p["comparison"]["rows"] if r["sample"]=="fixed_cohort" and r["period"]=="all")) for p in s["periods"]]),ensure_ascii=False,indent=2))


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    require(sys.version_info[:2]==(3,13) and not sysconfig.get_config_var("Py_GIL_DISABLED"),"Regular CPython 3.13 required")
    {"--precheck":precheck,"--run":execute,"--verify":verify}[sys.argv[1]]()
