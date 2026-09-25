"""Recalculate clustered ratio estimates from preserved independent replications."""
import json
import numpy as np
from simulation import summarize_values
from paths import SIMULATION, STATE

def refresh(directory, summaries):
    for path in directory.glob('des_*.json'):
        s=json.loads(path.read_text())
        data=directory/f'des_{s["name"]}_replications.npz'
        if not data.exists():
            continue
        values=np.load(data)['values']
        assert values.shape==(10000,22)
        s['statistics']=summarize_values(values)
        s['delay_estimator']='ratio of totals; independent-replication cluster delta-method t interval'
        path.write_text(json.dumps(s,indent=2))
    for name in summaries:
        path=directory/name
        if path.exists():
            rows=json.loads(path.read_text())
            rows=[json.loads((directory/f'des_{s["name"]}.json').read_text()) for s in rows]
            path.write_text(json.dumps(rows,indent=2))

if __name__=='__main__':
    refresh(SIMULATION,['simulation.json','validation_all.json'])
    refresh(STATE,['state_validation.json'])
    print('Refreshed statistics in simulation and state outputs from saved replications.')
