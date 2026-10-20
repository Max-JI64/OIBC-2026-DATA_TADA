"""D2: exact cohort/cancellation accounting on the same 178 V1 target hours."""
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
from validation_core import KEY, WINDOWS, local_score, metadata, require, times, truth_matrix, unique, weights
from v1_residual_diagnostics import frozen_artifacts

OUT = ROOT/"data/validation/2026-10-20_D2_B2_yearend"
REPORT = ROOT/"reports/2026-10-20_04_B2_연말_전체집합_악화분해.md"
B0 = ROOT/"data/validation/2026-10-19_B0_V1"
B2 = ROOT/"data/validation/2026-10-20_B2_joint"


def decomposition(fixed, rest, q):
    fixed,rest,q = (np.asarray(v,dtype=float) for v in (fixed,rest,q))
    require(fixed.ndim == rest.ndim == 2 and fixed.shape[0] == rest.shape[0] == len(q), "Cohort axis mismatch")
    require(len(q)>0 and fixed.shape[1]>0 and rest.shape[1]>0,"Empty cohort/time")
    require(np.isfinite(fixed).all() and np.isfinite(rest).all() and np.isfinite(q).all() and (q>=0).all(),"Invalid values")
    ef,er=fixed.sum(axis=1),rest.sum(axis=1)
    all_error=ef+er
    wf,wr,wa=[float(np.sum(np.abs(e)*q)) for e in (ef,er,all_error)]
    af=float(np.sum(np.abs(fixed)*q[:,None]))
    ar=float(np.sum(np.abs(rest)*q[:,None]))
    cross=float(np.sum((np.abs(ef)+np.abs(er)-np.abs(all_error))*q))
    require(cross>=-1e-5 and af+1e-5>=wf and ar+1e-5>=wr,"Negative cancellation")
    require(np.isclose(wa,wf+wr-cross,rtol=1e-12,atol=1e-5),"Cohort W identity failed")
    require(np.isclose(wa,af+ar-(af-wf)-(ar-wr)-cross,rtol=1e-12,atol=1e-5),"Individual W identity failed")
    result=dict(times=len(q),price_k_sum=float(q.sum()),W_fixed=wf,W_rest=wr,W_all=wa,
                abs_individual_fixed=af,abs_individual_rest=ar,
                within_fixed=af-wf,within_rest=ar-wr,cross_cancellation=cross)
    for label,e in (("fixed",ef),("rest",er),("all",all_error)):
        result[f"weighted_bias_{label}"]=float(np.sum(e*q)/q.sum()) if q.sum() else None
        result[f"over_W_{label}"]=float(np.sum(np.maximum(e,0)*q))
        result[f"under_W_{label}"]=float(np.sum(np.maximum(-e,0)*q))
        require(np.isclose(result[f"over_W_{label}"]+result[f"under_W_{label}"],result[f"W_{label}"],
                           rtol=1e-12,atol=1e-5),"Direction identity failed")
    return result


def delta(a,b):
    require(a["times"]==b["times"] and a["price_k_sum"]==b["price_k_sum"],"Different score sample")
    keys=("W_fixed","W_rest","W_all","abs_individual_fixed","abs_individual_rest",
          "within_fixed","within_rest","cross_cancellation")
    out={"delta_"+k:b[k]-a[k] for k in keys}
    require(np.isclose(out["delta_W_all"],out["delta_W_fixed"]+out["delta_W_rest"]-out["delta_cross_cancellation"],
                       rtol=1e-11,atol=1e-5),"Delta W identity failed")
    out["W_change_pct"]=(b["W_all"]/a["W_all"]-1)*100 if a["W_all"] else None
    return out


def protect():
    hashes=frozen_artifacts()
    files=[B2/"summary.json",B2/"precheck.json",B2/"V1/B2/predictions_mean.parquet",B2/"V1/B2/model_joint.txt",
           B2/"V1/B2/predictions_joint.parquet",B2/"T0/B2/predictions_mean.parquet",B2/"T0/B2/model_joint.txt",
           ROOT/"data/validation/2026-10-20_D1_V1_residuals/summary.json"]
    hashes.update({str(p.relative_to(ROOT)):digest(p) for p in files})
    return hashes


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    require(sys.version_info[:2]==(3,13) and not sysconfig.get_config_var("Py_GIL_DISABLED"),"Regular CPython 3.13 required")
    suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"),pattern="test_yearend_decomposition.py")
    tests=unittest.TextTestRunner(verbosity=2).run(suite)
    require(tests.wasSuccessful(),"Decomposition contracts failed")
    require(not (OUT/"summary.json").exists(),"Completed diagnosis exists")
    OUT.mkdir(parents=True,exist_ok=True)
    started=now()
    frozen=protect()
    sources={f:digest(SOURCE/f) for f in INPUTS}
    codes=[Path(__file__),ROOT/"scripts/validation_core.py",ROOT/"scripts/s0_validation_check.py",
           ROOT/"scripts/v1_residual_diagnostics.py",ROOT/"tests/test_yearend_decomposition.py"]
    codehash={str(p.relative_to(ROOT)):digest(p) for p in codes}
    (OUT/"executed_code").mkdir(exist_ok=True)
    for p in codes:
        (OUT/"executed_code"/p.name).write_bytes(p.read_bytes())
    b0=json.loads((B0/"summary.json").read_text(encoding="utf-8"))
    b2=json.loads((B2/"summary.json").read_text(encoding="utf-8"))
    require(b0["status"]==b2["status"]=="passed" and sources==b0["source_sha256"],"Previous run/source mismatch")
    sites=pd.read_csv(SOURCE/"sites.csv",encoding="utf-8-sig")
    metadata(sites)
    ids=sites.site_id.tolist()
    fixed=b0["fixed_148_sites"]
    rest=[i for i in ids if i not in fixed]
    require(len(fixed)==148 and len(rest)==105,"Cohort size changed")
    groups=sites.set_index("site_id").current_group.map({"A":"G1","B":"G2"})
    gen=pd.read_parquet(SOURCE/"train/generation.parquet")
    gen["ts"]=times(gen.ts)
    train=WINDOWS[0].training(gen)
    train_ids=set(train.site_id.unique())
    truth=truth_matrix(gen,ids,WINDOWS[0].axis())
    axis=truth.index[truth.notna().all(axis=1)]
    require(len(axis)==178,"Fixed common time count changed")
    price=pd.read_csv(SOURCE/"train/smp.csv",encoding="utf-8-sig")
    price["ts"]=times(price.ts)
    unique(price,["ts"])
    q=price.set_index("ts").loc[axis].smp.to_numpy()*weights(axis)
    require(np.isfinite(q).all() and (q>=0).all(),"Invalid prices")
    matrices={}
    score_rows=[]
    for model,path in (("B0",B0/"predictions_mean.parquet"),("B2",B2/"V1/B2/predictions_mean.parquet")):
        flat=pd.read_parquet(path)
        unique(flat,KEY)
        require(len(flat)==2222352,"Full prediction grid changed")
        matrix=flat.pivot(index="ts",columns="site_id",values="kwh").reindex(index=truth.index,columns=truth.columns)
        require(np.isfinite(matrix.to_numpy()).all() and matrix.ge(0).all().all(),"Invalid predictions")
        matrices[model]=matrix
        for name,members in (("fixed_148",fixed),("rest_105",rest),("all_253",ids)):
            score=local_score(truth[members],matrix[members],groups,price,axis)
            err=(matrix.loc[axis,members]-truth.loc[axis,members]).to_numpy()
            caps=sites.set_index("site_id").loc[members].capacity_kw.to_numpy()
            score_rows.append(dict(model=model,cohort=name,sites=len(members),cf_mae=float(np.abs(err/caps).mean()),
                                   kwh_mae=float(np.abs(err).mean()),**score))
        ref=(b0["group_scores"] if model=="B0" else b2["periods"][0]["joint_scores"])
        old=next(r for r in ref if r["sample"]=="common_253" and r["period"]=="all")
        got=score_rows[-1]
        for key in ("W_local","R_local","S_local","price_k_sum","times"):
            require(np.isclose(old[key],got[key],rtol=1e-12,atol=1e-5),"Original common score not reproduced")
    cohort_metadata=[]
    for name,members in (("fixed_148",fixed),("rest_105",rest)):
        for g in ("G1","G2"):
            part=[i for i in members if groups[i]==g]
            require(len(part)>0,"Missing group in cohort")
            cohort_metadata.append(dict(cohort=name,group=g,sites=len(part),
                capacity_kw=float(sites.set_index("site_id").loc[part].capacity_kw.sum()),
                has_train_sites=len(set(part)&train_ids),no_train_sites=len(set(part)-train_ids)))
    masks=[("all",np.ones(len(axis),dtype=bool))]+[(f"k={k}",weights(axis)==k) for k in (1,3)]
    days=(axis-pd.Timedelta(hours=1)).strftime("%Y-%m-%d")
    masks += [(str(day),days==day) for day in sorted(set(days))]
    rows,delta_rows=[],[]
    for g in ("G1","G2"):
        fids=[i for i in fixed if groups[i]==g]
        rids=[i for i in rest if groups[i]==g]
        for period,mask in masks:
            if not mask.any():
                continue
            values={}
            for model,matrix in matrices.items():
                e=matrix.loc[axis]-truth.loc[axis]
                metrics=decomposition(e[fids].to_numpy()[mask],e[rids].to_numpy()[mask],q[mask])
                rows.append(dict(group=g,period=period,model=model,**metrics))
                values[model]=metrics
            delta_rows.append(dict(group=g,period=period,**delta(values["B0"],values["B2"])))
    # Pool group W and cancellation amounts, never pool signed group residuals first.
    for period,_ in masks:
        model_values={}
        for model in ("B0","B2"):
            parts=[r for r in rows if r["group"] in ("G1","G2") and r["period"]==period and r["model"]==model]
            if not parts:
                continue
            summable=("W_fixed","W_rest","W_all","abs_individual_fixed","abs_individual_rest",
                      "within_fixed","within_rest","cross_cancellation")
            pooled={k:sum(r[k] for r in parts) for k in summable}
            pooled.update(times=parts[0]["times"],price_k_sum=parts[0]["price_k_sum"])
            rows.append(dict(group="both",period=period,model=model,**pooled))
            model_values[model]=pooled
        if model_values:
            delta_rows.append(dict(group="both",period=period,**delta(model_values["B0"],model_values["B2"])))
    keys=("W_fixed","W_rest","W_all","abs_individual_fixed","abs_individual_rest","within_fixed","within_rest","cross_cancellation")
    for model in ("B0","B2"):
        for g in ("G1","G2","both"):
            total=next(r for r in rows if r["group"]==g and r["period"]=="all" and r["model"]==model)
            for prefix in ("k=","2024-"):
                parts=[r for r in rows if r["group"]==g and r["period"].startswith(prefix) and r["model"]==model]
                require(sum(r["times"] for r in parts)==178,"Partition time mismatch")
                for key in keys:
                    require(np.isclose(sum(r[key] for r in parts),total[key],rtol=1e-11,atol=1e-5),"Partition mismatch")
            if g=="both":
                score=next(r for r in score_rows if r["model"]==model and r["cohort"]=="all_253")
                require(np.isclose(total["W_all"],score["W_local"],rtol=1e-12,atol=1e-5),"Decomposition differs from local score")
    for name,data in (("cohort_scores",score_rows),("cohort_metadata",cohort_metadata),("decomposition",rows),("delta",delta_rows)):
        pd.DataFrame(data).to_csv(OUT/f"{name}.csv",index=False,encoding="utf-8-sig")
    require(frozen==protect(),"Previous results/EDA changed")
    require(sources=={f:digest(SOURCE/f) for f in INPUTS},"Source changed")
    require(codehash=={str(p.relative_to(ROOT)):digest(p) for p in codes},"Code changed during run")
    write_json(OUT/"summary.json",dict(status="passed",experiment="D2_yearend_common178",started_at=started,
        finished_at=now(),runtime=dict(executable=sys.executable,python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__),
        synthetic_tests=tests.testsRun,source_sha256=sources,protected_sha256=frozen,code_sha256=codehash,
        times=len(axis),first_ts=str(axis.min()),last_ts=str(axis.max()),price_k_sum=float(q.sum()),
        model_fit_count=0,model_prediction_count=0,V2_reevaluated=False,cohort_metadata=cohort_metadata,
        cohort_scores=score_rows,decomposition=rows,deltas=delta_rows))
    log("D2 completed; same 178 hours; original W reproduced; cancellation identities passed; no model/V2 execution")


if __name__=="__main__":
    main()
