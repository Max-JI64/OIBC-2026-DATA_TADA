"""Training-only weighting units, alignment, zero price and peak boundary tests."""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from b5_loss_weighting import training_weights,audit_weight


class LossWeights(unittest.TestCase):
    def setUp(self):
        self.ts=pd.date_range("2023-01-01 10:00",periods=2,freq="h",tz="Asia/Seoul")
        self.price=pd.DataFrame(dict(ts=self.ts,smp=[2.,2.]))

    def test_capacity_l1_equals_scaled_kwh_l1(self):
        cap=np.array([10.,30.]);error=np.array([.2,-.1])
        w,scale=training_weights(cap,self.ts,self.price,"B5")
        self.assertAlmostEqual(np.sum(w*np.abs(error))*scale,np.sum(np.abs(cap*error)))
        np.testing.assert_allclose(w,[.5,1.5])

    def test_capacity_price_peak_boundary_and_order(self):
        w,scale=training_weights([10.,10.],self.ts,self.price,"B6")
        np.testing.assert_allclose(w,[.5,1.5])
        self.assertEqual(scale,40.)
        repeated=self.ts[[1,0,1]]
        w,_=training_weights([10.,10.,10.],repeated,self.price,"B6")
        np.testing.assert_allclose(w,np.array([60.,20.,60.])/(140/3))

    def test_zero_price_keeps_zero_weight_row(self):
        self.price.loc[0,"smp"]=0
        w,_=training_weights([10.,10.],self.ts,self.price,"B6")
        np.testing.assert_array_equal(w,[0.,2.])

    def test_missing_or_invalid_weights_rejected(self):
        with self.assertRaises(ValueError):training_weights([0.,10.],self.ts,self.price,"B5")
        with self.assertRaises(ValueError):training_weights([10.,10.],self.ts,self.price.iloc[:1],"B6")
        self.price.smp=0
        with self.assertRaises(ValueError):training_weights([10.,10.],self.ts,self.price,"B6")

    def test_audit_clip_and_effective_count(self):
        r=audit_weight(np.array([0.,2.]),1.,np.array([-.1,1.3]),np.array(["A","B"]))
        self.assertEqual(r["effective_rows"],1.)
        self.assertEqual(r["clip_high_rows"],1)
        self.assertEqual(r["clip_weight_share"],1.)
        self.assertEqual(r["G2_weight_share"],1.)


if __name__=="__main__":unittest.main()
