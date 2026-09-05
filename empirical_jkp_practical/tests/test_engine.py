import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from engine import *

class TestCausalEngine(unittest.TestCase):
    def synthetic(self):
        rng=np.random.default_rng(23)
        dates=pd.date_range('1973-01-31',periods=420,freq=pd.offsets.MonthEnd())
        frame=pd.DataFrame(rng.normal(.002,.03,(420,4)),index=dates,columns=list('abcd'))
        meta=pd.DataFrame({'group':['one','one','two','two']},index=list('abcd'))
        states=make_state(frame,meta).dropna();frame=frame.loc[states.index]
        cal=states.loc[:CALIBRATION_END].to_numpy();mu=cal.mean(0);sd=cal.std(0)
        return StudyData(frame,states,meta,mu,sd,2.,pd.DataFrame())
    def test_budget(self):
        e=np.array([.1,.01,.001,0]);targets=np.array([.5,1,2,2.9]);lam,c=lambdas_for_complexity(e,targets)
        np.testing.assert_allclose((e[:,None]/(e[:,None]+lam)).sum(0),targets,atol=1e-7)
        self.assertTrue(np.all(np.diff(lam)<0))
    def test_future_shock_weights(self):
        d=self.synthetic();a=rolling(d,kernel(d),window=60,budgets=np.array([1,5]),diagnostics=True)
        shockdate=pd.Timestamp('2005-01-31');d.returns.loc[shockdate:]*=-8
        d.state=make_state(d.returns,d.metadata).reindex(d.returns.index)
        original=self.synthetic();d.state=d.state.fillna(original.state)
        b=rolling(d,kernel(d),window=60,budgets=np.array([1,5]),diagnostics=True)
        mask=a['dates']<=shockdate
        np.testing.assert_allclose(a['weights'][mask],b['weights'][mask],atol=1e-6)
    def test_truncation(self):
        d=self.synthetic();a=rolling(d,kernel(d),window=60,budgets=np.array([2]))
        d.returns=d.returns.loc[:'2004-12-31'];d.state=d.state.loc[d.returns.index]
        b=rolling(d,kernel(d),window=60,budgets=np.array([2]))
        np.testing.assert_allclose(a['returns'][:len(b['dates'])],b['returns'])
    def test_primal_dual(self):
        rng=np.random.default_rng(21);X=rng.normal(size=(9,5));l=.3
        beta=np.linalg.solve(X.T@X/9+l*np.eye(5),X.mean(0))
        dual=X.T@np.linalg.solve(X@X.T+9*l*np.eye(9),np.ones(9))
        np.testing.assert_allclose(beta,dual,rtol=1e-10,atol=1e-12)
    def test_source_fields(self):
        d=self.synthetic();r=rolling(d,kernel(d),window=60,budgets=np.array([2,5]),diagnostics=True)
        np.testing.assert_allclose(r['contributions'].sum(2),r['returns'],atol=1e-7)
        self.assertLessEqual(r['gross'].max(),SLEEVE_GROSS_CAP+1e-8)
    def test_no_test_selection(self):
        d=self.synthetic();r=rolling(d,kernel(d),window=60,budgets=np.array([1,5,20]));j=choose(r)
        r['returns'][r['dates']>VALIDATION_END]=np.random.default_rng(1).normal(size=r['returns'][r['dates']>VALIDATION_END].shape)*100
        self.assertEqual(j,choose(r))
    def test_state_lag(self):
        d=self.synthetic();date=d.returns.index[100];old=make_state(d.returns,d.metadata)
        d.returns.loc[date:]*=99;new=make_state(d.returns,d.metadata)
        np.testing.assert_allclose(old.loc[date],new.loc[date])
    def test_nested_rff(self):
        d=self.synthetic();k1=kernel(d,rff_p=16);k2=kernel(d,rff_p=64)
        np.testing.assert_allclose(k1.diagonal(),1,atol=1e-12)
        np.testing.assert_allclose(k2.diagonal(),1,atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(k2)[0],-1e-10)
    def test_scaling_matches_dense_risk(self):
        rng=np.random.default_rng(51);history=rng.normal(size=(120,9))*.02;raw=rng.normal(size=(9,3))*.05
        w,multiplier=scale_weights(raw,history);cov=np.cov(history,rowvar=False)
        cov=(1-COV_SHRINK)*cov+COV_SHRINK*np.diag(np.diag(cov))+1e-10*np.eye(9)
        expected=TARGET_VOL/np.sqrt(12*np.einsum('il,ij,jl->l',raw,cov,raw))
        expected=np.minimum(expected,SLEEVE_GROSS_CAP/np.abs(raw).sum(0))
        np.testing.assert_allclose(multiplier,expected,rtol=1e-10);np.testing.assert_allclose(w,raw*expected)
    def test_rff_determinism(self):
        d=self.synthetic();np.testing.assert_array_equal(kernel(d,rff_p=64),kernel(d,rff_p=64))
    def test_rank_cap(self):
        e=np.array([1.,.1,0.,0.]);l,c=lambdas_for_complexity(e,np.array([.5,3.,20.]))
        self.assertLess(c.max(),2.);np.testing.assert_allclose((e[:,None]/(e[:,None]+l)).sum(0),c,atol=1e-7)

if __name__=='__main__':unittest.main()
