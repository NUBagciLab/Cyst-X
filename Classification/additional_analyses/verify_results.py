"""Compare regenerated aggregates with the accepted manuscript result snapshot."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('output_dir',type=Path)
a=p.parse_args(); ref=Path(__file__).parent/'reference_results'
manifest=json.loads((a.output_dir/'run_manifest.json').read_text())
if manifest.get('bootstrap')!=1000 or manifest.get('seed')!=20261024:
    raise AssertionError('Reference verification requires bootstrap=1000 and seed=20261024')
checks={}
for name in ['principal_ci','paired_comparison','frozen_pathology','frozen_pathology_auc_ci','matched_auc']:
    got=pd.read_csv(a.output_dir/(name+'.csv')); expected=pd.read_csv(ref/(name+'.csv'))
    if name=='matched_auc': expected=expected[expected.task=='2-class'].reset_index(drop=True)
    if list(got.columns)!=list(expected.columns): raise AssertionError(f'{name}: missing, reordered or unexpected columns')
    if len(got)!=len(expected): raise AssertionError(f'{name}: row count differs')
    for col in got:
        if col not in expected: raise AssertionError(f'{name}: unexpected column {col}')
        if pd.api.types.is_numeric_dtype(got[col]):
            np.testing.assert_allclose(got[col],expected[col],rtol=0,atol=1e-10,equal_nan=True,err_msg=f'{name}.{col}')
        elif not got[col].fillna('').equals(expected[col].fillna('')):
            raise AssertionError(f'{name}.{col}: values/order differ')
    checks[name]=dict(rows=len(got),status='PASS')
print(json.dumps(checks,indent=2))
