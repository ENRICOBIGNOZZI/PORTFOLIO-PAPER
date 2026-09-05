import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from core import *
from recommend import recommend

class Tests(unittest.TestCase):
    def setUp(self):self.R=np.random.default_rng(74).normal(.02,1.,(120,9))
    def test_nesting(self):
        f=fit_extension(self.R,3);base=np.linalg.lstsq(self.R[:,:3],np.ones(120),rcond=None)[0]
        np.testing.assert_allclose(f.weights[:3,0],base);np.testing.assert_array_equal(f.weights[3:,0],0)
    def test_profiled_normal_equations(self):
        f=fit_extension(self.R,3)
        for k in range(1,len(BUDGETS)):
            P=np.diag(np.r_[np.zeros(3),np.ones(6)])
            full=np.linalg.solve(self.R.T@self.R/120+f.ridge[k]*P,self.R.mean(0))
            np.testing.assert_allclose(f.weights[:,k],full,rtol=1e-8,atol=1e-10)
    def test_projection(self):
        f=fit_extension(self.R,3);H=self.R[:,3:]-self.R[:,:3]@f.hedge
        np.testing.assert_allclose(H.T@self.R[:,:3],0,atol=1e-12)
    def test_complexity_trace(self):
        f=fit_extension(self.R,3);B=self.R[:,:3];H=self.R[:,3:]-B@f.hedge
        for k in range(1,len(BUDGETS)):
            hat=B@np.linalg.solve(B.T@B,B.T)+H@np.linalg.solve(H.T@H+120*f.ridge[k]*np.eye(6),H.T)
            self.assertAlmostEqual(np.trace(hat),f.complexity[k],places=7)
    def test_in_sample_loss_monotone(self):
        f=fit_extension(self.R,3);q=((1-self.R@f.weights)**2).mean(0)
        self.assertTrue(np.all(np.diff(q)<=1e-10))
    def test_rank_cap(self):
        R=np.column_stack([self.R[:,:4],self.R[:,3]])
        f=fit_extension(R,3);self.assertLessEqual(f.complexity.max(),3.98+1e-7)
    def test_bad_input(self):
        with self.assertRaises(ValueError):fit_extension(np.full((10,5),np.nan),3)
    def test_baseline_rank(self):
        with self.assertRaises(ValueError):fit_extension(np.ones((30,5)),3)
    def test_gate_no_evidence(self):
        j,l=admit_extension(np.zeros((60,8)));self.assertEqual(j,-1)
    def test_gate_strong_evidence(self):
        x=np.zeros((60,3));x[:,1]=1.;j,l=admit_extension(x);self.assertEqual(j,1)
    def test_exact_oracle(self):
        m=np.array([.01,.005]);D=np.diag([.1,.2]);w=np.linalg.solve(D,m);q,sr=population_metrics(w,m,D)
        self.assertAlmostEqual(q[0],1-m@w);self.assertAlmostEqual(sr[0]**2/12,(1-q[0])/q[0])
    def test_spanning_identity(self):
        m=self.R.mean(0);S=np.cov(self.R,rowvar=False);beta=np.linalg.solve(S[:3,:3],S[:3,3:]);a=m[3:]-beta.T@m[:3]
        V=S[3:,3:]-S[3:,:3]@beta
        gain=m@np.linalg.solve(S,m)-m[:3]@np.linalg.solve(S[:3,:3],m[:3])
        self.assertAlmostEqual(gain,a@np.linalg.solve(V,a),places=10)
    def test_inference_identical(self):
        x=np.random.default_rng(42).normal(size=120);d=paired_inference(x,x,repetitions=50)
        for v in d.values():np.testing.assert_allclose(v,0,atol=1e-12)
    def test_online_future_invariance(self):
        rng=np.random.default_rng(19);dates=pd.date_range('1973-01-31',periods=600,freq=pd.offsets.MonthEnd())
        df=pd.DataFrame(rng.normal(.01,1,(600,5)),index=dates,columns=list('abcde'));meta=pd.DataFrame({'group':['base','base','base','g','g']},index=df.columns);rms=pd.Series(1.,index=df.columns)
        a=recommend(df,meta,rms,as_of='2010-12-31',window=60)
        df.loc['2011-01-01':]*=1000
        b=recommend(df,meta,rms,as_of='2010-12-31',window=60)
        self.assertEqual(a,b)
    def test_raw_scale_units(self):
        f=fit_extension(self.R,3);z=np.arange(1,10);raw=self.R*z
        np.testing.assert_allclose(raw@(f.weights/z[:,None]),self.R@f.weights,atol=1e-10)
    def test_determinism(self):
        a=fit_extension(self.R,3);b=fit_extension(self.R,3);np.testing.assert_array_equal(a.weights,b.weights)

if __name__=='__main__':unittest.main()
