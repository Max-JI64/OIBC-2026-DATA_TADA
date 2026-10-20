"""Past-only coefficient arithmetic and G1/k=3 correction isolation."""
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from b4_peak_calibration import weighted_median,adjust,direction


class PeakCalibration(unittest.TestCase):
    def test_weighted_median_and_tie(self):
        self.assertEqual(weighted_median([100,1,2],[0,1,1]),1)
        self.assertEqual(weighted_median([-2,0,5],[1,1,10]),5)
        with self.assertRaises(ValueError):weighted_median([1],[0])
        with self.assertRaises(ValueError):weighted_median([1],[-1])

    def test_correction_is_peak_g1_only_and_conserves(self):
        axis=pd.date_range("2024-01-01 10:00",periods=2,freq="h",tz="Asia/Seoul")
        frame=pd.DataFrame([[2.,3.,4.],[2.,3.,4.]],index=axis,columns=["a","b","c"])
        cap=pd.Series({"a":10.,"b":20.,"c":30.})
        out=adjust(frame,["a","b"],cap,-.1)
        pd.testing.assert_series_equal(out.iloc[0],frame.iloc[0])
        pd.testing.assert_series_equal(out.c,frame.c)
        self.assertAlmostEqual(out.iloc[1][["a","b"]].sum(),2)

    def test_zero_base_and_nonnegative_floor(self):
        axis=pd.date_range("2024-01-01 11:00",periods=1,freq="h",tz="Asia/Seoul")
        frame=pd.DataFrame([[0.,0.]],index=axis,columns=["a","b"])
        cap=pd.Series({"a":10.,"b":20.})
        np.testing.assert_allclose(adjust(frame,["a","b"],cap,.1).iloc[0],[1.,2.])
        self.assertEqual(adjust(frame,["a","b"],cap,-.1).to_numpy().sum(),0)

    def test_stability_zero_tolerance(self):
        self.assertEqual(direction(1e-9),0)
        self.assertEqual(direction(-.001),-1)


if __name__=="__main__":unittest.main()
