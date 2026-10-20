import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from b5_rolling_compare import complete_ids,decision,REPLAYS
from validation_core import Window,clock


class RollingContracts(unittest.TestCase):
    def pair(self,new=(9.,9.,9.)):
        base=[dict(sample="c",period=p,W_local=w,times=n,first_ts="a",last_ts="b",R_local=100.,price_k_sum=1.)
              for p,w,n in zip(("all","M07","M08","M09"),(30.,10.,10.,10.),(3,1,1,1))]
        other=[dict(r,W_local=w) for r,w in zip(base,(sum(new),*new))]
        return base,other

    def test_complete_cohort_keeps_real_zero(self):
        truth=pd.DataFrame({"a":[0.,1.],"b":[np.nan,2.],"c":[3.,4.]})
        self.assertEqual(complete_ids(truth),["a","c"])

    def test_both_cutoffs_precede_first_prediction(self):
        for w in REPLAYS:
            self.assertLess(pd.Timestamp(w.cutoff,tz="Asia/Seoul"),clock(w.axis()).prediction_at.min())
            gen=pd.DataFrame({"ts":pd.DatetimeIndex([pd.Timestamp(w.cutoff,tz="Asia/Seoul"),pd.Timestamp(w.cutoff,tz="Asia/Seoul")+pd.Timedelta(hours=1)])})
            self.assertEqual(len(w.training(gen)),1)

    def test_leaking_cutoff_rejected(self):
        w=Window("bad","2024-07-01","2024-10-01","2022-07-01","2024-07-01")
        with self.assertRaises(ValueError):w.training(pd.DataFrame({"ts":pd.DatetimeIndex(["2024-01-01"],tz="Asia/Seoul")}))

    def test_gate_checks_months_and_equal_total(self):
        self.assertTrue(decision(*self.pair(),"c")["passes"])
        self.assertFalse(decision(*self.pair((10.,10.,10.)),"c")["passes"])
        self.assertFalse(decision(*self.pair((10.5001,8.,8.)),"c")["passes"])
        self.assertTrue(decision(*self.pair((10.5,8.,8.)),"c")["passes"])

    def test_changed_score_sample_rejected(self):
        a,b=self.pair();b[0]["R_local"]=101.
        with self.assertRaises(ValueError):decision(a,b,"c")


if __name__=="__main__":unittest.main()
