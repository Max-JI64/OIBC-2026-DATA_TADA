"""H1 arithmetic, missing-data, time and strict decision contracts; no model fits."""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from h1_cloud_interaction import NEW, gate, h1_features
from validation_core import Window


def features(n):
    cols=["wa_cloud_r17","wb_cloud_r17","cosz","doy_sin","doy_cos"]+[f"old_{i}" for i in range(34)]
    x=pd.DataFrame(np.zeros((n,39),dtype="float32"),columns=cols)
    x["wa_cloud_r17"]=np.float32(40);x["wb_cloud_r17"]=np.float32(60)
    x["cosz"]=np.float32(.8);x["doy_sin"]=np.float32(.5);x["doy_cos"]=np.float32(-.5)
    return x


def scores(whole=100,period=100):
    return [dict(sample="fixed",period=p,times=1,first_ts="a",last_ts="b",R_local=1000,price_k_sum=1,W_local=w)
            for p,w in [("all",whole)]+[(f"Q{q}",period) for q in range(1,5)]]


class Contracts(unittest.TestCase):
    def test_exact_formula_and_unchanged_39_columns(self):
        x=features(2);y=h1_features(x)
        np.testing.assert_allclose(y[NEW].iloc[0],[.4,.2,-.2],rtol=1e-6)
        pd.testing.assert_frame_equal(y.iloc[:,:39],x)
        self.assertTrue(y.dtypes.eq("float32").all());self.assertEqual(y.shape,(2,42))

    def test_one_missing_or_invalid_cloud_not_filled(self):
        x=features(4);x["wa_cloud_r17"]=[np.nan,-1,101,np.inf]
        x=x.astype("float32");y=h1_features(x)
        self.assertTrue(y[NEW].isna().all().all())
        pd.testing.assert_frame_equal(y.iloc[:,:39],x)

    def test_night_and_finite_solar_contract(self):
        x=features(1);x["cosz"]=np.float32(-.5)
        self.assertTrue(h1_features(x)[NEW].eq(0).all().all())
        x["cosz"]=np.float32(np.nan)
        with self.assertRaises(ValueError):h1_features(x)

    def test_no_period_statistics_or_batch_dependence(self):
        a=features(1);b=pd.concat([a,features(20)],ignore_index=True);b.loc[1:,"wa_cloud_r17"]=100
        pd.testing.assert_series_equal(h1_features(a).iloc[0],h1_features(b).iloc[0])

    def test_five_percent_boundary_and_tie(self):
        self.assertTrue(gate(scores(),scores(90,105),"fixed",True)["passes"])
        self.assertFalse(gate(scores(),scores(90,105.001),"fixed",True)["passes"])
        self.assertFalse(gate(scores(),scores(100*(1-1e-13),90),"fixed",True)["passes"])

    def test_score_scope_mismatch_refused(self):
        a=scores();b=scores(90,90);b[0]["times"]=2
        with self.assertRaises(ValueError):gate(a,b,"fixed",True)

    def test_future_training_cutoff_refused(self):
        w=Window("bad","2024-01-01","2024-01-02","2022-01-01","2024-01-01")
        g=pd.DataFrame({"ts":pd.DatetimeIndex(["2023-01-01"],tz="Asia/Seoul")})
        with self.assertRaises(ValueError):w.training(g)


if __name__=="__main__":unittest.main(verbosity=2)
