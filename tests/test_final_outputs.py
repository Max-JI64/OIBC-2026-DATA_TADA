"""Reject invalid local submission grids and preserve training row order."""
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from b3_final_fit_predict import validate_outputs,combine_training_parts


class FinalOutputs(unittest.TestCase):
    def setUp(self):
        self.axis=pd.date_range("2025-06-01 01:00",periods=2,freq="h",tz="Asia/Seoul")
        self.sites=pd.DataFrame(dict(site_id=["a","b"],current_group=["A","B"],capacity_kw=[20000.,20000.]))
        self.price=pd.DataFrame(dict(ts=self.axis,smp=[0.,100.]))
        self.comp=pd.DataFrame(dict(site_id=["a","b"],group=["G1","G2"]))
        self.flat=pd.DataFrame(dict(site_id=["a","a","b","b"],ts=list(self.axis)*2,kwh=[1.,2.,3.,4.]))
        self.forecast=pd.DataFrame(dict(group=["G1","G1","G2","G2"],ts=[t.isoformat(sep=" ") for t in self.axis]*2,forecast_kwh=[1.,2.,3.,4.]))

    def check_output(self):return validate_outputs(self.comp,self.forecast,self.flat,self.sites,self.price)

    def test_complete_grid_keeps_zero_price_time(self):
        self.assertEqual(self.check_output()["zero_price_times_retained"],1)

    def test_missing_or_duplicate_group_timestamp(self):
        self.forecast=self.forecast.iloc[:-1]
        with self.assertRaises(ValueError):self.check_output()

    def test_missing_site_and_duplicate_composition(self):
        self.comp.loc[1,"site_id"]="a"
        with self.assertRaises(ValueError):self.check_output()

    def test_wrong_capacity_and_composition(self):
        self.sites.loc[0,"capacity_kw"]=19999.999
        with self.assertRaises(ValueError):self.check_output()

    def test_naive_timestamp_is_rejected(self):
        self.forecast.loc[0,"ts"]="2025-06-01 01:00:00"
        with self.assertRaises(ValueError):self.check_output()

    def test_invalid_value_at_zero_price_is_rejected(self):
        self.forecast.loc[0,"forecast_kwh"]=float("nan")
        with self.assertRaises(ValueError):self.check_output()

    def test_group_sum_must_match_sites(self):
        self.forecast.loc[0,"forecast_kwh"]=1.01
        with self.assertRaises(AssertionError):self.check_output()

    def test_monthly_assembly_restores_original_observation_order(self):
        xs=[pd.DataFrame({"x":np.array([30,10],dtype="float32")},index=[3,1]),pd.DataFrame({"x":np.array([20],dtype="float32")},index=[2])]
        ys=[pd.Series(np.array([3,1],dtype="float32"),index=[3,1]),pd.Series(np.array([2],dtype="float32"),index=[2])]
        x,y=combine_training_parts(xs,ys)
        np.testing.assert_array_equal(x.x,[10,20,30]);np.testing.assert_array_equal(y,[1,2,3])


if __name__=="__main__":unittest.main()
