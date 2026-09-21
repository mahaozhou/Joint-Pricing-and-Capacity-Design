"""Targeted AE staffing audit; retain every split and all registered cases.

Run from project root: python code/many_server_audit.py
No pathway/payment Cartesian products are added.
"""
import os
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
from dataclasses import asdict, replace
from concurrent.futures import ProcessPoolExecutor, as_completed
import json, csv, time
import numpy as np
from model import Model, optimize, evaluate, policy

from paths import STAFFING as OUT

def cases():
    rows = []
    for demand in (572., 878.):
        for d2 in (.1, .4, .7):
            rows.append((f'L{demand:g}_d{d2:g}', 'dual', Model(demand=demand, premium=15., d2=d2, servers=18)))
    for label, m in [('A', Model(demand=572., premium=15., d2=0., servers=18)),
                     ('B', Model(demand=878., premium=30., d2=.1, servers=18))]:
        for regime in ('fixed', 'online', 'dual'):
            rows.append((label, regime, m))
    return rows

def solve(task):
    label, regime, md, k = task
    m = Model(**md)
    start = time.perf_counter()
    r = optimize(m, regime, seed=41, iterations=450, restarts=6, fixed_capacity=k)
    q, e, *_ = m.matrices()
    loads = q @ np.array([r['flow_on'], r['flow_off']]) + e*r['required_actual']
    r.update(k_v=k, k_f=m.servers-k, load_v=float(loads[0]), load_f=float(loads[1]),
             rho_v=float(loads[0]/r['mu_on']), rho_f=float(loads[1]/r['mu_off']))
    assert r['residual'] < 1e-5 and max(r['rho_v'], r['rho_f']) < 1
    return dict(label=label, regime=regime, model=md, result=r, seconds=time.perf_counter()-start)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(label, regime, asdict(m), k) for label, regime, m in cases() for k in range(1,18)]
    (OUT/'design.json').write_text(json.dumps(tasks, indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        pending={}
        for task in tasks:
            name=f'{task[0]}_{task[1]}_k{task[3]}.json'
            path=OUT/name
            if path.exists(): rows.append(json.loads(path.read_text()))
            else: pending[pool.submit(solve,task)]=path
        for i,f in enumerate(as_completed(pending),1):
            r=f.result(); pending[f].write_text(json.dumps(r,indent=2));rows.append(r)
            if i%17==0: print(f'{i}/{len(pending)} splits completed',flush=True)
    selected=[]
    for label,regime,m in cases():
        rr=[r for r in rows if r['label']==label and r['regime']==regime]
        best=max(r['result']['objective'] for r in rr)
        selected.append(min((r for r in rr if r['result']['objective']>=best-2e-4),key=lambda r:r['result']['exposure']))
    (OUT/'selected.json').write_text(json.dumps(selected,indent=2))
    fields=['label','regime','demand','d2','k_v','k_f','p_on','p_off','objective','access','load_v','load_f','rho_v','rho_f','residual']
    with (OUT/'utilization.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in selected:
            w.writerow({key: (r[key] if key in ('label','regime') else r['model'][key] if key in ('demand','d2') else r['result'][key]) for key in fields})
    for r in selected:
        z=r['result']; print(r['label'],r['regime'],z['k_v'],z['k_f'],round(z['objective'],3),round(z['rho_v'],4),round(z['rho_f'],4),flush=True)

if __name__=='__main__': main()
