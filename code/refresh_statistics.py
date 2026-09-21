"""Recalculate clustered ratio estimates from preserved independent replications."""
import json
import numpy as np
from simulation import OUT,summarize_values

for path in OUT.glob('des_*.json'):
    s=json.loads(path.read_text());data=OUT/f'des_{s["name"]}_replications.npz'
    if not data.exists():continue
    values=np.load(data)['values']
    assert values.shape==(10000,22)
    s['statistics']=summarize_values(values)
    s['delay_estimator']='ratio of totals; independent-replication cluster delta-method t interval'
    path.write_text(json.dumps(s,indent=2))
for name in ['simulation.json','validation_all.json','state_validation.json']:
    path=OUT/name
    if path.exists():
        rows=json.loads(path.read_text())
        rows=[json.loads((OUT/f'des_{s["name"]}.json').read_text()) for s in rows]
        path.write_text(json.dumps(rows,indent=2))
print('Recomputed final statistics from 10,000-replication arrays; no simulation observations changed.')
