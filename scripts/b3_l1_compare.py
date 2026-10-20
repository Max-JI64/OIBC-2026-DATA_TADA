"""Two fixed B3 fits; objective-only ablation against frozen B2/B0 predictions."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import sysconfig
import unittest

import numpy as np
import pandas as pd

from b2_joint_compare import OUT as B2, B0, T0, fit_predict, scores, compare, runtime
from b0_v1_baseline import model_parameters
from b2_yearend_decomposition import decomposition, protect
from s0_validation_check import ROOT,SOURCE,INPUTS,baseline_contract,digest,load_forecasts,log,now,write_json
from validation_core import WINDOWS,KEY,times,truth_matrix,unique,metadata,require,weights

OUT=ROOT/"data/validation/2026-10-20_B3_L1"
REPORT=ROOT/"reports/2026-10-20_05_B3_L1손실_비교.md"


def parameters():
    return {**model_parameters(),"objective":"regression_l1"}


def protected():
    result=protect()
    for window,methods in (("V1",("B2",)),("T0",("B0","B2"))):
        for method in methods:
            folder=B2/window/method
            for path in folder.glob("*"):
                if path.is_file():
                    result[str(path.relative_to(ROOT))]=digest(path)
    return result


def code_files():
    return [Path(__file__),ROOT/"scripts/b2_joint_compare.py",ROOT/"scripts/b0_v1_baseline.py",
            ROOT/"scripts/b2_yearend_decomposition.py",ROOT/"scripts/v1_residual_diagnostics.py",
            ROOT/"scripts/validation_core.py",ROOT/"scripts/s0_validation_check.py",ROOT/"tests/test_b3_validation.py"]


def read_prediction(path,truth):
    flat=pd.read_parquet(path)
    unique(flat,KEY)
    require(len(flat)==len(truth)*253,"Serialized prediction count mismatch")
    mat=flat.pivot(index="ts",columns="site_id",values="kwh").reindex(index=truth.index,columns=truth.columns)
    require(np.isfinite(mat.to_numpy()).all() and mat.ge(0).all().all(),"Invalid predictions")
    return mat


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    require(sys.version_info[:2]==(3,13) and not sysconfig.get_config_var("Py_GIL_DISABLED"),"Regular CPython 3.13 required")
    suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"),pattern="test_*validation.py")
    tests=unittest.TextTestRunner(verbosity=2).run(suite)
    require(tests.wasSuccessful(),"B3 preflight failed")
    b2=json.loads((B2/"summary.json").read_text(encoding="utf-8"))
    oldpre=json.loads((B2/"precheck.json").read_text(encoding="utf-8"))
    require(b2["status"]=="passed" and b2["runtime"]==runtime() and b2["model_parameters"]==model_parameters(),"B2 environment/settings changed")
    if "--precheck" in sys.argv:
        require(not (OUT/"precheck.json").exists(),"Precheck already frozen")
        OUT.mkdir(parents=True,exist_ok=True)
        source={f:digest(SOURCE/f) for f in INPUTS}
        require(source==oldpre["source_sha256"],"Source drift")
        code={str(p.relative_to(ROOT)):digest(p) for p in code_files()}
        (OUT/"executed_code").mkdir(exist_ok=True)
        for p in code_files():
            (OUT/"executed_code"/p.name).write_bytes(p.read_bytes())
        (OUT/"preregistered_report.md").write_bytes(REPORT.read_bytes())
        write_json(OUT/"precheck.json",dict(at=now(),runtime=runtime(),source_sha256=source,protected_sha256=protected(),
            code_sha256=code,prereg_report_sha256=digest(REPORT),parameters=parameters(),fit_budget=2,
            model_fit_count=0,scores_computed=False))
        log("B3 precheck passed; only objective changes; fit budget=2; no score/fit")
        return
    require(not (OUT/"summary.json").exists(),"Completed B3 run exists")
    pre=json.loads((OUT/"precheck.json").read_text(encoding="utf-8"))
    require(pre["runtime"]==runtime() and pre["parameters"]==parameters() and digest(REPORT)==pre["prereg_report_sha256"],"Precheck changed")
    require(pre["protected_sha256"]==protected(),"Previous artifacts changed")
    require(pre["source_sha256"]=={f:digest(SOURCE/f) for f in INPUTS},"Source changed")
    require(pre["code_sha256"]=={str(p.relative_to(ROOT)):digest(p) for p in code_files()},"Code changed")
    started=now()
    sites=pd.read_csv(SOURCE/"sites.csv",encoding="utf-8-sig")
    metadata(sites)
    groups=sites.set_index("site_id").current_group.map({"A":"G1","B":"G2"})
    gen=pd.read_parquet(SOURCE/"train/generation.parquet");gen["ts"]=times(gen.ts);unique(gen,KEY)
    price=pd.read_csv(SOURCE/"train/smp.csv",encoding="utf-8-sig");price["ts"]=times(price.ts);unique(price,["ts"])
    spec=baseline_contract()
    fa,fb,excluded=load_forecasts("train",spec)
    outputs=[];yearend=[]
    for index,window in enumerate((WINDOWS[0],T0)):
        truth=truth_matrix(gen,sites.site_id,window.axis())
        ids=oldpre["V1_cohort_ids"] if index==0 else oldpre["T0"]["cohort_ids"]
        cohort=b2["periods"][index]["cohort_name"]
        matrices={"B2":read_prediction(B2/window.name/"B2/predictions_mean.parquet",truth)}
        path=B0/"predictions_mean.parquet" if index==0 else B2/"T0/B0/predictions_mean.parquet"
        matrices["B0"]=read_prediction(path,truth)
        old_scores={}
        for method in ("B0","B2"):
            old_scores[method]=scores(truth,matrices[method],groups,price,ids,cohort)
            ref=b2["periods"][index]["baseline_scores" if method=="B0" else "joint_scores"]
            refmap={(r["sample"],r["period"]):r for r in ref}
            for item in old_scores[method]:
                for key in ("W_local","R_local","S_local","price_k_sum","times"):
                    require(np.isclose(item[key],refmap[(item["sample"],item["period"])][key],rtol=1e-12,atol=1e-5),"Frozen score mismatch")
        matrices["B3"],fits=fit_predict(window,"B3",gen,sites,fa,fb,spec,parameters(),truth,output_root=OUT)
        require(len(fits)==1 and fits[0]["features"]==next(f for f in b2["periods"][index]["fits"] if f["mode"]=="joint")["features"],"Feature drift")
        b2fit=next(f for f in b2["periods"][index]["fits"] if f["mode"]=="joint")
        for key in ("train_rows","daylight_rows","train_ts_min","train_ts_max","any_feature_nan_rows"):
            require(fits[0][key]==b2fit[key],"Training sample changed")
        new=scores(truth,matrices["B3"],groups,price,ids,cohort)
        decision=compare(old_scores["B2"],new,cohort)
        against0=compare(old_scores["B0"],new,cohort)
        outputs.append(dict(window=vars(window),cohort_name=cohort,cohort_sites=len(ids),fits=fits,
            B0_scores=old_scores["B0"],B2_scores=old_scores["B2"],B3_scores=new,vs_B2=decision,vs_B0=against0))
        if index==0:
            axis=truth.index[truth.notna().all(axis=1)]
            require(len(axis)==178,"Common sample changed")
            rest=[i for i in truth.columns if i not in ids]
            q=price.set_index("ts").loc[axis].smp.to_numpy()*weights(axis)
            for method,mat in matrices.items():
                error=mat.loc[axis]-truth.loc[axis]
                for g in ("G1","G2"):
                    fids=[i for i in ids if groups[i]==g];rids=[i for i in rest if groups[i]==g]
                    yearend.append(dict(method=method,group=g,**decomposition(error[fids],error[rids],q)))
        log(f"{window.name} B3 comparison complete")
    require(sum(len(p["fits"]) for p in outputs)==2,"Fit budget exceeded")
    require(pre["protected_sha256"]==protected(),"Previous artifacts/EDA changed")
    require(pre["source_sha256"]=={f:digest(SOURCE/f) for f in INPUTS},"Source changed")
    require(pre["code_sha256"]=={str(p.relative_to(ROOT)):digest(p) for p in code_files()},"Code changed during run")
    passes=all(p["vs_B2"]["passes"] for p in outputs)
    write_json(OUT/"summary.json",dict(status="passed",experiment="B3_L1_only",started_at=started,finished_at=now(),
        runtime=runtime(),parameters=parameters(),precheck_sha256=digest(OUT/"precheck.json"),synthetic_tests=tests.testsRun,
        model_fit_count=2,V2_executed=False,final_evaluation_executed=False,periods=outputs,yearend=yearend,
        selection="B3_development_pass" if passes else "B3_not_adopted",nonhourly=excluded))
    pd.DataFrame(yearend).to_csv(OUT/"yearend.csv",index=False,encoding="utf-8-sig")
    log(f"B3 fixed run passed; fits=2; development_pass={passes}; no V2")


if __name__=="__main__":
    main()
