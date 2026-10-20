"""One frozen B3 final fit, separate train/test forecasts, local outputs only."""
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

from b0_v1_baseline import prediction_values, target_values
from b2_joint_compare import frames, joint_features, runtime
from b3_l1_compare import parameters
from b3_final_design_check import FINAL
from s0_validation_check import ROOT, SOURCE, INPUTS, baseline_contract, digest, load_forecasts, log, monthly, now, write_json
from validation_core import KEY, clock, grid, metadata, require, times, unique, weights

OUT=ROOT/"data/final/2026-10-20_B3"
P1=ROOT/"data/validation/2026-10-20_B3_final_design"
B3=ROOT/"data/validation/2026-10-20_B3_L1"
B4=ROOT/"data/validation/2026-10-20_B4_peak_calibration"
REPORT=ROOT/"reports/2026-10-20_09_B3_최종학습과_평가예측_생성.md"


def combine_training_parts(xs,ys):
    x=pd.concat(xs).sort_index(kind="stable")
    y=pd.concat(ys)
    require(x.index.is_unique and y.index.is_unique and set(x.index)==set(y.index),"Training order keys mismatch")
    y=y.reindex(x.index)
    require(x.dtypes.eq(np.dtype("float32")).all() and y.dtype==np.dtype("float32"),"Training dtype changed")
    require(np.isfinite(y).all() and not np.isinf(x.to_numpy()).any(),"Invalid assembled training values")
    return x.reset_index(drop=True),y.reset_index(drop=True)


def validate_site_grid(flat,sites,axis):
    require(flat.columns.tolist()==KEY+["kwh"],"Unexpected site prediction schema")
    unique(flat,KEY)
    require(len(flat)==len(sites)*len(axis) and set(flat.site_id)==set(sites.site_id),"Incomplete site grid")
    ts=times(flat.ts)
    require(ts.isin(axis).all() and ts.equals(ts.floor("h")),"Site time axis mismatch")
    require(np.isfinite(flat.kwh).all() and flat.kwh.ge(0).all(),"Invalid site predictions")


def validate_outputs(composition,forecast,flat,sites,price):
    require(composition.columns.tolist()==["site_id","group"],"Unexpected composition schema")
    unique(composition,["site_id"])
    require(len(composition)==len(sites) and set(composition.site_id)==set(sites.site_id),"Composition coverage mismatch")
    require(set(composition.group)=={"G1","G2"},"Invalid output groups")
    expected=sites.set_index("site_id").current_group.map({"A":"G1","B":"G2"})
    require(composition.set_index("site_id").group.reindex(expected.index).equals(expected.rename("group")),"Reference composition changed")
    capacities=composition.merge(sites[["site_id","capacity_kw"]],on="site_id",validate="one_to_one").groupby("group").capacity_kw.sum()
    require(capacities.ge(20000).all(),"Output capacity constraint failed")
    axis=times(price.ts);unique(price,["ts"])
    require(axis.is_unique and axis.equals(axis.floor("h")),"Bad price axis")
    validate_site_grid(flat,sites,axis)
    require(forecast.columns.tolist()==["group","ts","forecast_kwh"],"Unexpected forecast schema")
    require(forecast.ts.map(lambda v:isinstance(v,str) and v.endswith("+09:00")).all(),"Explicit KST output timestamps required")
    checked=forecast.copy();checked["ts"]=times(checked.ts)
    unique(checked,["group","ts"])
    require(set(checked.group)=={"G1","G2"} and len(checked)==2*len(axis),"Incomplete group grid")
    require(np.isfinite(checked.forecast_kwh).all() and checked.forecast_kwh.ge(0).all(),"Invalid group predictions")
    mat=flat.pivot(index="ts",columns="site_id",values="kwh").reindex(index=axis,columns=sites.site_id)
    require(np.isfinite(mat.to_numpy()).all(),"Missing site prediction cells")
    largest=0.
    for g in ("G1","G2"):
        part=checked.loc[checked.group.eq(g)]
        require(times(part.ts).equals(axis),"Group target times differ from price axis")
        members=composition.loc[composition.group.eq(g),"site_id"]
        direct=np.sum(mat.loc[:,members].to_numpy(),axis=1)
        saved=part.forecast_kwh.to_numpy()
        np.testing.assert_allclose(saved,direct,rtol=1e-12,atol=1e-7)
        largest=max(largest,float(np.max(np.abs(saved-direct))))
    all_groups=checked.pivot(index="ts",columns="group",values="forecast_kwh").reindex(axis).sum(axis=1).to_numpy()
    np.testing.assert_allclose(all_groups,np.sum(mat.to_numpy(),axis=1),rtol=1e-12,atol=1e-7)
    return dict(site_rows=len(flat),composition_rows=len(composition),forecast_rows=len(forecast),times=len(axis),
                capacity_kw={str(k):float(v) for k,v in capacities.items()},max_group_sum_abs_difference=largest,
                zero_price_times_retained=int(price.smp.eq(0).sum()))


def protected_inputs():
    pre=json.loads((B4/"precheck.json").read_text(encoding="utf-8"))
    protected=dict(pre["protected_sha256"])
    for p in B4.rglob("*"):
        if p.is_file():protected[str(p.relative_to(ROOT))]=digest(p)
    for p in (ROOT/"reports/2026-10-20_08_B4_G1피크_과거잔차_보정비교.md",):
        protected[str(p.relative_to(ROOT))]=digest(p)
    require(all(digest(ROOT/p)==h for p,h in protected.items()),"Previous artifacts changed")
    source={f:digest(SOURCE/f) for f in INPUTS}
    require(source==pre["source_sha256"],"Sources changed")
    return source,protected


def code_paths():
    return [Path(__file__),ROOT/"tests/test_final_outputs.py",ROOT/"tests/test_b2_validation.py",ROOT/"tests/test_b3_validation.py",ROOT/"scripts/b0_v1_baseline.py",
            ROOT/"scripts/b2_joint_compare.py",ROOT/"scripts/b3_l1_compare.py",ROOT/"scripts/b3_final_design_check.py",
            ROOT/"scripts/validation_core.py",ROOT/"scripts/s0_validation_check.py"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    require(sys.version_info[:2]==(3,13) and not sysconfig.get_config_var("Py_GIL_DISABLED"),"Regular CPython 3.13 required")
    suite=unittest.TestSuite()
    for pattern in ("test_final_outputs.py","test_b2_validation.py","test_b3_validation.py"):
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/"tests"),pattern=pattern))
    tests=unittest.TextTestRunner(verbosity=2).run(suite);require(tests.wasSuccessful(),"Final contracts failed")
    design=json.loads((P1/"summary.json").read_text(encoding="utf-8"))
    final_contract=json.loads((P1/"finalized_contract.json").read_text(encoding="utf-8"))
    old=json.loads((B3/"summary.json").read_text(encoding="utf-8"))
    b4=json.loads((B4/"summary.json").read_text(encoding="utf-8"))
    require(b4["status"]=="passed" and b4["selection"]=="retain_B3","B3 no-calibration selection changed")
    require(design["window"]==vars(FINAL) and design["model_parameters"]==old["parameters"]==parameters(),"Final design/settings changed")
    require(digest(P1/"summary.json")==final_contract["availability_summary_sha256"],"Design availability changed")
    require(digest(ROOT/"reports/2026-10-20_06_B3_최종학습과_평가절차_설계.md")==final_contract["report_sha256"],"Final design report changed")
    require(runtime()==old["runtime"],"B3 runtime changed")
    source,protected=protected_inputs()
    if "--precheck" in sys.argv:
        require(not (OUT/"precheck.json").exists(),"Precheck already frozen")
        OUT.mkdir(parents=True,exist_ok=True);(OUT/"executed_code").mkdir(exist_ok=True)
        for p in code_paths():(OUT/"executed_code"/p.name).write_bytes(p.read_bytes())
        (OUT/"preregistered_report.md").write_bytes(REPORT.read_bytes())
        write_json(OUT/"precheck.json",dict(at=now(),runtime=runtime(),source_sha256=source,protected_sha256=protected,
            code_sha256={str(p.relative_to(ROOT)):digest(p) for p in code_paths()},prereg_report_sha256=digest(REPORT),
            window=vars(FINAL),parameters=parameters(),features=design["features"],model_fit_count=0,
            model_fit_budget=1,performance_scores_computed=False,synthetic_tests=tests.testsRun))
        log("Final precheck passed; B3 frozen; model fits/predictions/scores=0")
        return
    require(not (OUT/"summary.json").exists(),"Completed final run exists")
    pre=json.loads((OUT/"precheck.json").read_text(encoding="utf-8"))
    require(source==pre["source_sha256"] and protected==pre["protected_sha256"] and digest(REPORT)==pre["prereg_report_sha256"],"Precheck drift")
    require(pre["code_sha256"]=={str(p.relative_to(ROOT)):digest(p) for p in code_paths()},"Code changed after precheck")
    started=now();spec=baseline_contract()
    sites=pd.read_csv(SOURCE/"sites.csv",encoding="utf-8-sig");capacities=metadata(sites)
    price=pd.read_csv(SOURCE/"smp.csv",encoding="utf-8-sig");price["ts"]=times(price.ts);unique(price,["ts"])
    axis=FINAL.axis();require(times(price.ts).equals(axis) and len(axis)==8760,"Final price axis changed")
    require(np.array_equal(price.k.to_numpy(),weights(axis)),"Final k changed")
    require(np.isfinite(price.smp).all() and price.smp.ge(0).all() and price.smp.eq(0).sum()==18,"Final prices changed")
    model_path=OUT/"model_joint.txt"
    if "--resume-predictions" in sys.argv:
        fit=json.loads((OUT/"fit_summary.json").read_text(encoding="utf-8"))
        require(digest(model_path)==fit["model_sha256"] and fit["features"]==pre["features"],"Saved model changed")
        require(fit["parameters"]==pre["parameters"] and fit["daylight_rows"]==design["train"]["daylight_rows"],"Saved fit settings changed")
        model=lgb.Booster(model_str=model_path.read_text(encoding="utf-8"))
        require(model.num_trees()==600,"Saved tree count mismatch")
    else:
        require(not model_path.exists() and not (OUT/"fit_journal.json").exists(),"Model already attempted; never refit automatically")
        gen=pd.read_parquet(SOURCE/"train/generation.parquet");gen["ts"]=times(gen.ts);unique(gen,KEY)
        require(gen.site_id.isin(sites.site_id).all() and np.isfinite(gen.energy).all(),"Invalid training truth")
        require(not gen.ts.isin(axis).any(),"Evaluation truth unexpectedly present")
        train=FINAL.training(gen).reset_index(drop=True);train["_train_order"]=np.arange(len(train),dtype=np.int64)
        require(len(train)==design["train"]["rows"] and train.site_id.nunique()==253,"Final train rows/sites mismatch")
        require(str(train.ts.min())==design["train"]["first_ts"] and str(train.ts.max())==design["train"]["last_ts"],"Final train bounds changed")
        require(train.ts.max()<clock(axis).prediction_at.min(),"Training availability leakage")
        del gen
        fa,fb,train_nonhourly=load_forecasts("train",spec)
        xs=[];ys=[];observed=0
        for batch in monthly(pd.DatetimeIndex(train.ts.unique()).sort_values()):
            base=train.loc[train.ts.isin(batch)].merge(sites,on="site_id",how="left",validate="many_to_one",sort=False)
            d,x10,d17,x17=frames(base,fa,fb,spec);x=joint_features(d,x10,d17,x17,spec)
            require(x.columns.tolist()==pre["features"],"Joint feature order mismatch")
            day=d.elev.gt(0);order=d.loc[day,"_train_order"].to_numpy()
            part=x.loc[day].copy();part.index=order
            target=target_values(d.loc[day]);target.index=order
            xs.append(part);ys.append(target);observed+=len(d)
            del base,d,x10,d17,x17,x,part,target
            log(f"Final training features: {batch[0]:%Y-%m}")
        require(observed==len(train),"Training monthly coverage changed")
        x,y=combine_training_parts(xs,ys)
        require(len(x)==design["train"]["daylight_rows"] and x.shape[1]==39,"Daylight training shape changed")
        nan_rows=int(x.isna().any(axis=1).sum())
        del xs,ys,train,fa,fb;gc.collect()
        write_json(OUT/"fit_journal.json",dict(started_at=now(),model_fit_count=1,status="started"))
        log(f"Final B3: fit {x.shape}; original observation order restored")
        tick=time.perf_counter();estimator=lgb.LGBMRegressor(**parameters()).fit(x,y)
        require(estimator.booster_.num_trees()==600,"Final tree count mismatch")
        model_path.write_text(estimator.booster_.model_to_string(),encoding="utf-8")
        fit=dict(train_rows=observed,daylight_rows=len(x),features=x.columns.tolist(),any_feature_nan_rows=nan_rows,
            train_ts_min=design["train"]["first_ts"],train_ts_max=design["train"]["last_ts"],parameters=parameters(),
            fit_seconds=time.perf_counter()-tick,model_sha256=digest(model_path),train_nonhourly=train_nonhourly,
            original_observation_order_restored=True,model_fit_count=1)
        write_json(OUT/"fit_summary.json",fit)
        write_json(OUT/"fit_journal.json",dict(finished_at=now(),model_fit_count=1,status="saved",model_sha256=digest(model_path)))
        del estimator,x,y;gc.collect()
        model=lgb.Booster(model_str=model_path.read_text(encoding="utf-8"))
    require(model.feature_name()==pre["features"] and model.dump_model()["objective"]=="regression_l1","Saved final model contract mismatch")
    fa,fb,test_nonhourly=load_forecasts("test",spec)
    (OUT/"months").mkdir(exist_ok=True)
    pieces=[];monthly_info=[];reused=0
    for batch in monthly(axis):
        name=f"{batch[0]:%Y-%m}";file=OUT/"months"/f"{name}.parquet";info_path=file.with_suffix(".json")
        if file.exists():
            info=json.loads(info_path.read_text(encoding="utf-8"))
            require(info["sha256"]==digest(file) and info["model_sha256"]==fit["model_sha256"],"Partial monthly output drift")
            flat=pd.read_parquet(file);validate_site_grid(flat,sites,batch);reused+=1
        else:
            base=grid(sites,batch);d,x10,d17,x17=frames(base,fa,fb,spec);x=joint_features(d,x10,d17,x17,spec)
            require(x.columns.tolist()==pre["features"],"Inference feature order mismatch")
            values=prediction_values(d,model.predict(x))
            require((values[d.elev.to_numpy()<=0]==0).all(),"Night predictions changed")
            flat=d[KEY].assign(kwh=values);validate_site_grid(flat,sites,batch)
            flat.to_parquet(file,index=False,compression="zstd")
            info=dict(month=name,rows=len(flat),feature_nan_rows=int(x.isna().any(axis=1).sum()),
                first_ts=str(batch[0]),last_ts=str(batch[-1]),model_sha256=fit["model_sha256"],sha256=digest(file))
            write_json(info_path,info)
            del base,d,x10,d17,x17,x,values
        pieces.append(flat);monthly_info.append(info)
        log(f"Final prediction: {name}, rows={len(flat):,}")
    del model,fa,fb;gc.collect()
    flat=pd.concat(pieces,ignore_index=True);validate_site_grid(flat,sites,axis)
    flat.to_parquet(OUT/"site_predictions.parquet",index=False,compression="zstd")
    composition=sites[["site_id"]].assign(group=sites.current_group.map({"A":"G1","B":"G2"}))
    composition.to_csv(OUT/"composition.csv",index=False,encoding="utf-8-sig")
    grouped=flat.assign(group=flat.site_id.map(composition.set_index("site_id").group)).groupby(["group","ts"],as_index=False).kwh.sum()
    forecast=grouped.rename(columns={"kwh":"forecast_kwh"}).sort_values(["group","ts"]).reset_index(drop=True)
    forecast.ts=forecast.ts.map(lambda v:v.isoformat(sep=" "))
    forecast.to_csv(OUT/"forecast.csv",index=False,encoding="utf-8-sig")
    saved_flat=pd.read_parquet(OUT/"site_predictions.parquet")
    pd.testing.assert_frame_equal(saved_flat,flat)
    saved_composition=pd.read_csv(OUT/"composition.csv",encoding="utf-8-sig")
    saved_forecast=pd.read_csv(OUT/"forecast.csv",encoding="utf-8-sig",float_precision="round_trip")
    check=validate_outputs(saved_composition,saved_forecast,saved_flat,sites,price)
    require(all((OUT/name).read_bytes().startswith(b"\xef\xbb\xbf") for name in ("composition.csv","forecast.csv")),"CSV BOM missing")
    source2,protected2=protected_inputs();require(source==source2 and protected==protected2,"Input drift during final execution")
    require(pre["code_sha256"]=={str(p.relative_to(ROOT)):digest(p) for p in code_paths()},"Code changed during final run")
    require(digest(REPORT)==pre["prereg_report_sha256"],"Execution scope changed")
    outputs={name:digest(OUT/name) for name in ("composition.csv","forecast.csv","site_predictions.parquet","model_joint.txt","fit_summary.json")}
    write_json(OUT/"summary.json",dict(status="passed",started_at=started,finished_at=now(),runtime=runtime(),
        precheck_sha256=digest(OUT/"precheck.json"),chosen_method="B3_no_calibration",window=vars(FINAL),parameters=parameters(),
        fit=fit,monthly_predictions=monthly_info,output_validation=check,output_sha256=outputs,
        model_fit_count=1,synthetic_tests=tests.testsRun,reused_prediction_months=reused,test_nonhourly=test_nonhourly,
        source_unchanged=True,protected_unchanged=True,performance_scores_computed=False,V2_executed=False,
        official_submission_executed=False,calibration_applied=False))
    log("Final B3 passed: 1 fit, local composition/forecast/site outputs verified; no performance score or submission")


if __name__=="__main__":main()
