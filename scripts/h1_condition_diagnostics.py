"""Replay the frozen EDA-011 conditions on saved H1/B3 predictions; no fits/inference."""
from __future__ import annotations
import gc
from pathlib import Path
import sys
import sysconfig
import numpy as np
import pandas as pd
from h1_cloud_interaction import OUT, WINDOWS, inputs, load_prediction, read
from forecast_residual_eda import OUT as EDA, FAMILIES, conditions, forecasts, group_conditions, cohort_ids, pred_path
from s0_validation_check import ROOT, baseline_contract, digest, now, write_json
from validation_core import clock, grid, require, times, truth_matrix, weights


def freeze():
    require(not (OUT/"summary.json").exists() and not any(OUT.glob("*/fit_journal.json")),"Freeze diagnostic code before fitting")
    p=OUT/"condition_contract.json"
    require(not p.exists(),"Condition contract already frozen")
    write_json(p,dict(at=now(),code_sha256=digest(Path(__file__)),families=FAMILIES,
                      eda_code_sha256=digest(ROOT/"scripts/forecast_residual_eda.py"),
                      fit_started=False,threshold_changes=0))
    print("EDA condition diagnostic code frozen before H1 fits")


def cloud_contrast(d,labels):
    z=d.loc[d.main].copy();z["condition"]=labels.loc[z.index,"cloud17"]
    z["solar_bin"]=pd.cut(z.elev,[15,30,50,np.inf],right=False).astype("string")
    keys=["month","hour","solar_bin"]
    a=z.loc[z.condition.eq("cloudy>80")];b=z.loc[~z.condition.isin(["cloudy>80","missing"])]
    def cells(q):
        v=q.groupby(keys,observed=True).agg(cap=("capacity_kw","sum"),actual=("energy","sum"),
                                         B3=("p3","sum"),H1=("ph","sum"),days=("date","nunique"))
        for col in ("actual","B3","H1"):v[col]/=v.cap
        return v.loc[v.days.ge(5)]
    ca,cb=cells(a),cells(b);ix=ca.index.intersection(cb.index)
    if not len(ix):return None
    delta=ca.loc[ix,["actual","B3","H1"]]-cb.loc[ix,["actual","B3","H1"]]
    return dict(cells=len(ix),delta_actual=float(delta.actual.mean()),delta_B3=float(delta.B3.mean()),
                delta_H1=float(delta.H1.mean()),B3_bias_contrast=float((delta.B3-delta.actual).mean()),
                H1_bias_contrast=float((delta.H1-delta.actual).mean()))


def main():
    if "--freeze" in sys.argv:freeze();return
    pre=read(OUT/"precheck.json");contract=read(OUT/"condition_contract.json");summary=read(OUT/"summary.json")
    require(contract["code_sha256"]==digest(Path(__file__)) and contract["eda_code_sha256"]==digest(ROOT/"scripts/forecast_residual_eda.py"),"Diagnostic code changed")
    require(summary["status"]=="passed","Completed model results required")
    sites,gen,price=inputs();spec=baseline_contract();fixed=set(cohort_ids("V1"))
    old=pd.read_csv(EDA/"group_conditions.csv",encoding="utf-8-sig")
    oldm=pd.read_csv(EDA/"matched_contrasts.csv",encoding="utf-8-sig")
    rows=[];contrasts=[];contributions=[];partitions=0
    for w,av in zip(WINDOWS,pre["periods"]):
        axis=w.axis();ids=av["cohort_ids"]
        truth=truth_matrix(gen,sites.site_id,axis)
        oldpred=load_prediction(pred_path(w.name,"B3"),truth)
        newpred=load_prediction(OUT/w.name/"predictions.parquet",truth)
        d=grid(sites.loc[sites.site_id.isin(ids)],axis)
        d["energy"]=truth.loc[axis,d.site_id.unique()].T.to_numpy().reshape(-1)
        d["p3"]=oldpred.loc[axis,d.site_id.unique()].T.to_numpy().reshape(-1)
        d["ph"]=newpred.loc[axis,d.site_id.unique()].T.to_numpy().reshape(-1)
        # grid is site-major by construction; verify independent keyed reindex.
        idx=pd.MultiIndex.from_frame(d[["site_id","ts"]])
        check=truth.rename_axis(index="ts",columns="site_id").stack().swaplevel().reindex(idx)
        require(np.array_equal(check.to_numpy(),d.energy.to_numpy()),"Condition truth key order differs")
        fs,qa=forecasts(axis)
        for station,q in fs:d=d.merge(q,on=[station,"ts"],how="left",validate="many_to_one",sort=False)
        require(len(d)==len(ids)*len(axis),"Condition join expanded")
        t=clock(d.ts);d["date"]=t.target_day.to_numpy();d["month"]=t.start.dt.month.to_numpy();d["hour"]=t.start.dt.hour.to_numpy()
        d["elev"]=spec["solar"](times(d.ts),d.lat.to_numpy(),d.lon.to_numpy())[0]
        d["main"]=d.hour.between(9,16)&d.elev.ge(15);d["group"]=d.current_group.map({"A":"G1","B":"G2"})
        d["membership"]=np.where(d.site_id.isin(fixed),"fixed_148","added_complete")
        d["e3"]=d.p3-d.energy;d["eh"]=d.ph-d.energy
        pp=price.set_index("ts").smp.reindex(axis);q=pp*weights(axis)
        d["q"]=d.ts.map(q);labels=conditions(d)
        for method,err in (("B3","e3"),("H1","eh")):
            e=d.groupby(["group","ts"])[err].transform("sum")
            d[method+"_contribution"]=d.q*np.sign(e)*d[err]
        for name,members in (("complete",ids),("fixed_148",[i for i in ids if i in fixed]),
                              ("added_complete",[i for i in ids if i not in fixed])):
            if not members:continue
            for group in ("G1","G2"):
                z=d.loc[d.site_id.isin(members)&d.group.eq(group)]
                if len(z)==0:continue
                labs=group_conditions(z)
                a=z.groupby("ts")[["energy","p3","ph","capacity_kw"]].sum();a["q"]=q
                total={method:float((q*(a[p]-a.energy).abs()).sum()) for method,p in (("B3","p3"),("H1","ph"))}
                for family in FAMILIES:
                    rr=[]
                    for label,ix in labs.groupby(family).groups.items():
                        v=a.loc[ix];r=dict(window=w.name,cohort=name,group=group,family=family,condition=label,
                                          times=len(v),days=clock(v.index).target_day.nunique(),
                                          actual_cf=float(v.energy.sum()/v.capacity_kw.sum()))
                        for method,p in (("B3","p3"),("H1","ph")):
                            err=v[p]-v.energy
                            r[method+"_cf"]=float(v[p].sum()/v.capacity_kw.sum())
                            r[method+"_bias"]=float(err.sum()/v.capacity_kw.sum())
                            r[method+"_W"]=float((v.q*err.abs()).sum())
                            r[method+"_over_W"]=float((v.q*err.clip(lower=0)).sum())
                            r[method+"_under_W"]=float((v.q*(-err).clip(lower=0)).sum())
                        r["change_pct"]=(r["H1_W"]/r["B3_W"]-1)*100 if r["B3_W"] else None
                        o=old.loc[old.window.eq(w.name)&old.cohort.eq(name)&old.group.eq(group)&old.family.eq(family)&old.condition.eq(label)]
                        require(len(o)==1 and int(o.iloc[0].times)==len(v),"EDA condition support changed")
                        for field in ("B3_W","B3_over_W","B3_under_W","actual_cf","B3_cf","B3_bias"):
                            require(np.isclose(r[field],o.iloc[0][field],rtol=1e-11,atol=1e-10),"EDA B3 condition values differ")
                        rr.append(r);rows.append(r)
                    for method in ("B3","H1"):
                        require(np.isclose(sum(r[method+"_W"] for r in rr),total[method],rtol=1e-12),"Condition W not conserved")
                        require(all(np.isclose(r[method+"_W"],r[method+"_over_W"]+r[method+"_under_W"],rtol=1e-12) for r in rr),"Direction split differs")
                        if name=="complete":partitions+=1
                con=cloud_contrast(z,labels)
                if con:
                    om=oldm.loc[oldm.window.eq(w.name)&oldm.cohort.eq(name)&oldm.group.eq(group)&oldm.family.eq("cloud17")&oldm.condition.eq("cloudy>80")].iloc[0]
                    require(con["cells"]==om.cells and np.isclose(con["B3_bias_contrast"],om.delta_bias,rtol=1e-11,atol=1e-12),"EDA matched contrast differs")
                    contrasts.append(dict(window=w.name,cohort=name,group=group,**con))
        for family in FAMILIES:
            cond=labels[family].where(d.main,"outside_main")
            cr=d.assign(condition=cond).groupby(["group","membership","condition"])[["B3_contribution","H1_contribution"]].sum().reset_index()
            contributions.extend([dict(window=w.name,family=family,**r) for r in cr.to_dict("records")])
            for method,err in (("B3","e3"),("H1","eh")):
                whole=d.groupby(["group","ts"]).agg(err=(err,"sum"),q=("q","first"))
                require(np.isclose(cr[method+"_contribution"].sum(),(whole.q*whole.err.abs()).sum(),rtol=1e-12),"Site weather signed contribution failed")
        del d,z,truth,oldpred,newpred;gc.collect()
    for name,value in (("condition_comparison",rows),("cloud_contrasts",contrasts),("signed_contributions",contributions)):
        pd.DataFrame(value).to_csv(OUT/(name+".csv"),index=False,encoding="utf-8-sig")
    require(partitions==128,"Expected 4 windows x 2 groups x 8 families x 2 methods")
    write_json(OUT/"condition_verification.json",dict(at=now(),status="passed",group_partitions=partitions,
                                                   rows=len(rows),cloud_contrasts=len(contrasts),new_fits=0,new_inference=0,
                                                   EDA_B3_conditions_reproduced=True,code_sha256=digest(Path(__file__)),
                                                   outputs_sha256={p.name:digest(p) for p in OUT.glob("*.csv")}))
    print(f"H1 condition diagnostics passed: {len(rows)} rows, {partitions} group partitions, {len(contrasts)} cloud contrasts")


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    require(sys.version_info[:2]==(3,13) and not sysconfig.get_config_var("Py_GIL_DISABLED"),"Regular CPython 3.13 required")
    main()
