"""Validate generated tables and the archived-run narrative before publication."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

def verify(out: Path) -> None:
    models=pd.read_csv(out/'selected_models.csv').set_index('model')
    returns=pd.read_csv(out/'all_selected_OOS_returns.csv',index_col=0)
    ranking=pd.read_csv(out/'feature_ranking_VALIDATION_ONLY.csv')
    weights=pd.read_csv(out/'selected_factor_weights.csv',index_col=0)
    contribution=pd.read_csv(out/'oos_spectral_contributions.csv',index_col=0)
    assert returns.shape==(252,47),returns.shape
    assert returns.notna().all().all()
    assert len(ranking)==153 and ranking.characteristic.nunique()==153
    assert ranking.validation_contribution_sr.is_monotonic_decreasing
    assert returns.index[0]=='2005-01-31' and returns.index[-1]=='2025-12-31'
    for name in returns:
        r=returns[name].to_numpy()
        np.testing.assert_allclose(np.sqrt(12)*r.mean()/r.std(ddof=1),models.loc[name,'sharpe'],rtol=1e-8,atol=1e-9)
    np.testing.assert_allclose(contribution.sum(axis=1),returns.matern_full,rtol=1e-6,atol=1e-8)
    assert weights.abs().sum(axis=1).max()<=3.00001
    chosen=pd.read_csv(out/'VALIDATION_SELECTED_compact_model.csv').iloc[0]
    candidates=models.loc[['compact_top8','compact_top16','compact_top32','compact_top64','matern_full']]
    assert chosen['model']==candidates.validation_sharpe.idxmax()
    # Human interpretation in the report refers to this archived experiment.
    # A changed vintage must trigger review, not silently keep its old narrative.
    reference={'matern_full':.6983418032,'constant_full':1.243263,'compact_top8':.495162,'compact_top16':.712263,'matern_ew':1.575433,'published_by2004':.568274}
    for name,value in reference.items():
        if not np.isclose(models.loc[name,'sharpe'],value,rtol=0,atol=2e-5):
            raise RuntimeError(f'Revised result for {name}; review the narrative before publication: {models.loc[name,"sharpe"]}')
    audit={'status':'passed','selected_return_series_reconciled':len(returns.columns),'test_months':len(returns),'unique_feature_ablations':len(ranking),'spectral_contributions_reconcile':True,'factor_sleeve_cap_respected':True,'compact_choice_uses_validation':True,'archived_narrative_snapshot_verified':True}
    (out/'RESULTS_VERIFICATION.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('empirical_jkp_practical/results'));a=p.parse_args();verify(a.out)
