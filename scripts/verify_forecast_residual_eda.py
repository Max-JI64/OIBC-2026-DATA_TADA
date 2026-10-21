"""Check EDA-011 tables independently of the calculation loop; no models."""
from __future__ import annotations

import sys
import sysconfig
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_residual_eda import OUT, read
from s0_validation_check import digest, now, write_json
from validation_core import require


def main():
    require(sys.version_info[:2]==(3,13) and not sysconfig.get_config_var("Py_GIL_DISABLED"),"Regular CPython 3.13 required")
    g=pd.read_csv(OUT/"group_conditions.csv",encoding="utf-8-sig")
    s=pd.read_csv(OUT/"site_conditions.csv",encoding="utf-8-sig")
    m=pd.read_csv(OUT/"matched_contrasts.csv",encoding="utf-8-sig")
    c=pd.read_csv(OUT/"signed_contributions.csv",encoding="utf-8-sig")
    b=pd.read_csv(OUT/"daily_blocks.csv",encoding="utf-8-sig")
    a=pd.read_csv(OUT/"audits.csv",encoding="utf-8-sig")
    checks=0
    for table in (g,s):
        for method in ("B3","B5"):
            require(np.allclose(table[method+"_cf"]-table.actual_cf,table[method+"_bias"],rtol=1e-11,atol=1e-13),"CF/bias identity failed")
            checks+=1
    for method in ("B3","B5"):
        require((s[method+"_site_abs"]+1e-12>=s[method+"_bias"].abs()).all(),"Absolute error below bias magnitude")
        checks+=1
    require(s.small.eq((s.days<20)|(s.stationsA<5)|(s.stationsB<5)).all(),"Small support flags differ")
    require(b.small.eq(b.active_days<20).all(),"Daily support flags differ")
    require(np.allclose(m.delta_pred-m.delta_actual,m.delta_bias,rtol=1e-10,atol=1e-12),"Matched actual/prediction/error identity failed")
    checks+=3
    # A condition family partitions group-time, unlike site-specific subset W.
    base=g.loc[g.family.eq("cloud17")].groupby(["window","cohort","group"])[["B3_W","B5_W","times"]].sum()
    for key,z in a.loc[a.dimension.notna()].groupby(["window","cohort","group","dimension"]):
        target=base.loc[key[:3]]
        require(z.times.sum()==target.times,"Month/hour/k times not conserved")
        for method in ("B3","B5"):
            require(np.isclose(z[method+"_W"].sum(),target[method+"_W"],rtol=1e-12),"Month/hour/k W not conserved")
        checks+=1
    full=base.loc[(slice(None),"complete",slice(None)),:].droplevel("cohort")
    for key,z in c.groupby(["window","group","family"]):
        target=full.loc[key[:2]]
        for method in ("B3","B5"):
            require(np.isclose(z[method+"_contribution"].sum(),target[method+"_W"],rtol=1e-12),"Group signed contributions do not sum to W")
        checks+=1
    join=a.loc[a.joined_rows.notna()]
    require(len(join)==16,"Expected four joins per window")
    require(join.unmatched_rows.ge(0).all() and join.matched_rain_nan.ge(0).all(),"Invalid missing audit")
    require((a.loc[a.column.notna(),"invalid"]==0).all(),"Out-of-range forecast values require explicit review")
    checks+=3
    pre=read(OUT/"precheck.json")
    require(digest(OUT/"executed_code/forecast_residual_eda.py")==pre["code_sha256"],"Frozen executed code mismatch")
    write_json(OUT/"table_verification.json",dict(at=now(),status="passed",checks=checks,
                                                verifier_sha256=digest(Path(__file__)),csv_encoding="utf-8-sig",
                                                matched_rows=len(m),site_rows=len(s),group_rows=len(g),join_audits=len(join)))
    print(f"Independent EDA table checks passed: {checks}; matched={len(m)}, site={len(s)}, group={len(g)}")


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
