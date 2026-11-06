"""Reproduce the bundled scenarios offline: python examples/run_extended_matrix.py."""
import json
from pathlib import Path
from inferfit import UnsupportedModelError
from inferfit.cli import evaluate_document

root=Path(__file__).resolve().parent/'extended'
cases=json.loads((root/'scenarios.json').read_text())
rows=[]
for case in cases:
    try:
        result=evaluate_document(case['input'],mode=case['mode'],base=root)
    except UnsupportedModelError as error:
        result={'status':'unsupported','memory_pass':None,'reasons':[str(error)]}
    if result['status']!=case['expected_status']:
        raise RuntimeError((case['name'],case['expected_status'],result))
    summary=result.get('summary',{})
    rows.append({'name':case['name'],'mode':case['mode'],'status':result['status'],
                 'memory_pass':result['memory_pass'],
                 'minimum_headroom_bytes':summary.get('minimum_headroom_bytes'),
                 'kv_bytes_per_token_logical':summary.get('kv_bytes_per_token_logical'),
                 'reasons':result.get('reasons',[]),
                 'deployment_verified':False})
(root/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'scenarios':len(rows),'counts':{s:sum(r['status']==s for r in rows) for s in ['pass','fail','unsupported','invalid']}},indent=2))
