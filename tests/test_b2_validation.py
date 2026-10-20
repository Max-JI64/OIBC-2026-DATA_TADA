"""B2 joint-input alignment, missing-data preservation and comparison boundaries."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from b2_joint_compare import T0, compare, frames, joint_features
from s0_validation_check import baseline_contract
from validation_core import KST, clock


class JointContracts(unittest.TestCase):
    def setUp(self):
        self.spec=baseline_contract()
        self.d=pd.DataFrame({"site_id":["x","y"],"ts":pd.date_range("2024-01-01 11:00",periods=2,tz=KST,freq="h"),
                             "energy":[1.,2.],"cap":[10.,10.]})
        self.x=pd.DataFrame(np.ones((2,24),dtype="float32"),columns=self.spec["FEATS"])

    def test_39_features_keep_both_weather_rounds_and_pressure(self):
        other=self.x.copy()
        other.loc[0,"wa_temp"]=np.nan
        j=joint_features(self.d,self.x,self.d.copy(),other,self.spec)
        self.assertEqual(j.shape,(2,39))
        self.assertTrue(np.isnan(j.wa_temp_r17.iloc[0]))
        self.assertEqual(j.wa_temp_r10.iloc[0],1.)
        self.assertIn("wa_ground_press_r10",j)
        self.assertIn("wa_ground_press_r17",j)
        self.assertEqual(j.columns.tolist()[-9:],self.spec["FEATS"][15:])

    def test_order_or_target_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            joint_features(self.d,self.x,self.d.iloc[::-1],self.x,self.spec)
        d=self.d.copy();d.loc[0,"energy"]=5.
        with self.assertRaises(AssertionError):
            joint_features(self.d,self.x,d,self.x,self.spec)

    def test_missing_forecast_rows_still_keep_joint_grid(self):
        base=self.d.assign(current_group="A",capacity_kw=10.,lat=35.,lon=127.,wa_station="a",wb_station="b")
        fa=pd.DataFrame({"wa_station":["a"],"round":[10],"ts":[base.ts.iloc[0]],
                         **{c:[1.] for c in self.spec["WA_COLS"]}})
        fb=pd.DataFrame({"wb_station":["b"],"round":[10],"ts":[base.ts.iloc[0]],
                         **{c:[1.] for c in self.spec["WB_COLS"]}})
        d,x,d17,x17=frames(base,fa,fb,self.spec)
        joint=joint_features(d,x,d17,x17,self.spec)
        self.assertEqual(len(joint),2)
        self.assertTrue(joint.filter(like="_r17").isna().all().all())

    def test_t0_boundary_has_17_hour_gap(self):
        gen=pd.DataFrame({"ts":pd.DatetimeIndex(["2022-01-01 00:00","2022-01-01 01:00","2022-12-31 00:00","2022-12-31 01:00"],tz=KST)})
        self.assertEqual(T0.training(gen).ts.tolist(),gen.ts.iloc[[1,2]].tolist())
        self.assertEqual(len(T0.axis()),8760)
        self.assertEqual(clock(T0.axis()).prediction_at.min()-pd.Timestamp(T0.cutoff,tz=KST),pd.Timedelta(hours=17))

    def test_selection_and_different_truth_guard(self):
        def make(w):
            return [dict(sample="cohort",period=p,times=1,first_ts="a",last_ts="b",price_k_sum=1.,R_local=10.,
                         W_local=w,weighted_group_mae_kwh=w/2) for p in ("all","Q1","Q2","Q3","Q4")]
        self.assertTrue(compare(make(100.),make(99.),"cohort")["passes"])
        self.assertFalse(compare(make(100.),make(100.),"cohort")["passes"])
        b=make(99.);b[1]["W_local"]=105.01
        self.assertFalse(compare(make(100.),b,"cohort")["passes"])
        b=make(99.);b[0]["R_local"]=11.
        with self.assertRaises(ValueError):
            compare(make(100.),b,"cohort")


if __name__ == "__main__":
    unittest.main()
