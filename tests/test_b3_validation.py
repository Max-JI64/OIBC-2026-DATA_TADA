"""Objective-only B3 contract, retaining the existing frozen joint-input behavior."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from b3_l1_compare import parameters
from b0_v1_baseline import model_parameters
from b2_joint_compare import compare


class L1Contracts(unittest.TestCase):
    def test_only_objective_changes(self):
        new=parameters();old=model_parameters()
        self.assertEqual(new.pop("objective"),"regression_l1")
        self.assertEqual(new,old)
        self.assertNotIn("alpha",new)

    def test_improvement_and_quarter_guard(self):
        def rows(w):
            return [dict(sample="fixed",period=p,times=1,first_ts="a",last_ts="b",price_k_sum=1.,R_local=1.,
                W_local=w,weighted_group_mae_kwh=w/2) for p in ("all","Q1","Q2","Q3","Q4")]
        self.assertTrue(compare(rows(100),rows(99),"fixed")["passes"])
        self.assertFalse(compare(rows(100),rows(100),"fixed")["passes"])
        new=rows(99);new[1]["W_local"]=105.01
        self.assertFalse(compare(rows(100),new,"fixed")["passes"])


if __name__=="__main__":
    unittest.main()
