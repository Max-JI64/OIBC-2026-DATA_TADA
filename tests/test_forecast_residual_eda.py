"""Boundary and availability contracts for EDA-011, without data or models."""
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from forecast_residual_eda import conditions, group_conditions, block_interval


def frame(n):
    d=pd.DataFrame({f"{p}_{v}_r{r}":np.zeros(n) for p in ("wa","wb") for v in ("cloud","rain") for r in (10,17)})
    return d


class Contracts(unittest.TestCase):
    def test_missing_not_dry_or_clear(self):
        d=frame(1);d.loc[0,"wa_rain_r17"]=np.nan;d.loc[0,"wb_cloud_r17"]=np.nan
        c=conditions(d)
        self.assertEqual(c.cloud17.iloc[0],"missing")
        self.assertEqual(c.rain_provider.iloc[0],"missing")
        self.assertEqual(c.rainA17.iloc[0],"missing")

    def test_fixed_cloud_and_revision_boundaries(self):
        d=frame(4);d["wa_cloud_r10"]=40.;d["wa_cloud_r17"]=[20,60,61,100];d["wb_cloud_r17"]=[20,100,100,100]
        c=conditions(d)
        self.assertEqual(c.cloud17.tolist(),["clear<=20","mixed20-80","cloudy>80","cloudy>80"])
        self.assertEqual(c.revisionA.tolist(),["down<=-20","stable(-20,20]","up>20","up>20"])

    def test_group_coverage_and_capacity_occurrence(self):
        d=frame(2);d["ts"]=pd.Timestamp("2024-07-01 12:00",tz="Asia/Seoul");d["capacity_kw"]=[96.,4.];d["main"]=True
        d.loc[1,"wa_cloud_r17"]=np.nan;d.loc[0,"wa_rain_r17"]=1.
        c=group_conditions(d)
        self.assertEqual(c.cloud17.iloc[0],"clear<=20")
        self.assertEqual(c.rain_provider.iloc[0],"first_only")
        d["capacity_kw"]=[94.,6.]
        self.assertEqual(group_conditions(d).cloud17.iloc[0],"missing")

    def test_outside_main_is_not_dropped(self):
        d=frame(1);d["ts"]=pd.Timestamp("2024-07-01 01:00",tz="Asia/Seoul");d["capacity_kw"]=1.;d["main"]=False
        self.assertTrue(group_conditions(d).eq("outside_main").all().all())

    def test_temporal_block_constant_and_short_support(self):
        self.assertEqual(block_interval(np.ones(40)),(1.,1.))
        self.assertEqual(block_interval(np.ones(19)),(None,None))


if __name__=="__main__":
    unittest.main(verbosity=2)
