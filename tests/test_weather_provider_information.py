"""Synthetic checks for CCA, weighted calendar removal and circular cross-products."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from weather_provider_information import canonical, covariance, centered, month_lags, frozen_correlations, bootstrap_weights


class ProviderInformationTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(55)

    def test_identical_and_independent(self):
        x = self.rng.normal(size=(20000,6))
        y = np.column_stack([x,self.rng.normal(size=len(x))])
        fit = canonical(np.cov(np.column_stack([x,y]),rowvar=False,bias=True))
        np.testing.assert_allclose(fit['roots'],1,atol=1e-12)
        np.testing.assert_allclose(fit['remaining_a'],0,atol=1e-12)
        self.assertGreater(fit['remaining_b'][-1],.99)

    def test_rank_failure(self):
        x = self.rng.normal(size=(1000,13))
        x[:,1] = x[:,0]
        self.assertFalse(canonical(np.cov(x,rowvar=False))['valid'])

    def test_scale_invariance(self):
        z = self.rng.normal(size=(3000,13))
        z[:,6:12] += z[:,:6]
        a = canonical(np.cov(z,rowvar=False))
        b = canonical(np.cov(z*np.arange(1,14),rowvar=False))
        np.testing.assert_allclose(a['roots'],b['roots'],atol=1e-12)
        np.testing.assert_allclose(a['remaining_b'],b['remaining_b'],atol=1e-12)

    def test_weighted_absorption(self):
        z = self.rng.normal(size=(15,2,3,13))
        m = np.repeat([1,2,3],5)
        w = np.array([0,0,0,0,0,1,2,1,0,1,1,1,2,2,1])
        ix = np.repeat(np.arange(15),w)
        cz = centered(z[ix],m[ix]).reshape(-1,13)
        np.testing.assert_allclose(covariance(z,m,w),cz.T@cz/len(cz),atol=1e-12)

    def test_level_covariance(self):
        z = self.rng.normal(size=(10,2,3,13)) + np.arange(13)
        np.testing.assert_allclose(covariance(z,np.ones(10),within=False),np.cov(z.reshape(-1,13),rowvar=False,bias=True),atol=1e-12)

    def test_fft_all_lags(self):
        z = self.rng.normal(size=(11,2,3,13))
        m = np.array([1]*5+[2]*6)
        out = month_lags(z,m)
        for month in [1,2]:
            x = z[m==month].reshape(sum(m==month),-1,13)
            for lag in range(len(x)):
                direct = x[...,:6].reshape(-1,6).T@np.roll(x[...,6:],-lag,axis=0).reshape(-1,7)
                np.testing.assert_allclose(out[month][lag],direct,atol=1e-12)

    def test_frozen_correlations(self):
        x = self.rng.normal(size=(3000,13))
        x[:,6:12] += x[:,:6]
        cov = np.cov(x,rowvar=False,bias=True)
        fit = canonical(cov)
        np.testing.assert_allclose(frozen_correlations(cov,fit),fit['roots'],atol=1e-12)

    def test_joint_weights(self):
        w = bootstrap_weights(366,self.rng)
        self.assertEqual(w.sum(),366)
        self.assertEqual(len(w),366)


if __name__ == '__main__':
    unittest.main()
