"""Registered deterministic design. Reruns resume by content-addressed config.
Run: python revision/experiments.py --workers 4
"""
import os
os.environ.setdefault('NUMBA_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import json, hashlib, time, argparse
from dataclasses import asdict, replace
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from model import Model, optimize, evaluate, policy
from regional_solver import is_regional_baseline, optimize_regional

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/revision'
OUT.mkdir(parents=True,exist_ok=True)

def design():
    tasks=[]
    def add(group,label,m,regime='dual'):
        tasks.append(dict(group=group,label=label,model=asdict(m),regime=regime))
    # All grids declared before result inspection.
    for demand in [572.,878.]:
        for premium in [15.,30.]:
            for d2 in [0.,.1,.2,.35,.5,.65,.8]:
                for regime in ['fixed','online','dual']:
                    add('referral',f'L{demand:g}_s{premium:g}_d{d2:g}',Model(demand=demand,premium=premium,d2=d2),regime)
    for d1 in [.1,.2,.35,.5,.65,.8]:
        for d2 in [0.,.15,.3,.45,.6,.75]:
            add('map',f'd1_{d1:g}_d2_{d2:g}',Model(demand=878,premium=15,d1=d1,d2=d2))
    for req in [0.,30.,60.,90.]:
        for reg in ['fixed','online','dual']:
            for variant in ['total','residual','no_income']:
                m=Model(demand=878,premium=15,required=req,d2=.4)
                if variant=='residual': m=replace(m,capacity=972+(req-30)*1.4)
                if variant=='no_income': m=replace(m,revenue_E=0.)
                add('required',f'{variant}_E{req:g}',m,reg)
    for d2 in [.1,.4,.7]:
        for variant in ['one_referral','staged','recurrent']:
            m=Model(demand=878,premium=15,d2=d2)
            if variant=='one_referral':m=replace(m,gamma=0.)
            if variant=='recurrent':m=replace(m,recurrent=True)
            add('pathway',f'{variant}_d{d2:g}',m)
    for fee in [0.,.5,1.]:
        for copay in [.3,1.]:
            add('payment',f'fee{fee:g}_copay{copay:g}',Model(demand=878,premium=15,d2=.4,followup_fee=fee,copay=copay))
    for de in [0.,.1,.4,.7]:
        add('heterogeneity',f'dE{de:g}',Model(demand=878,premium=15,d2=.4,dE=de))
    for elasticity in [0.,.01,.03]:
        add('elasticity',f'eta{elasticity:g}',Model(demand=878,premium=15,d2=.4,elasticity=elasticity))
    for demand in [572.,878.]:
        for d2 in [.1,.4,.7]:
            for variant in ['staged','one_referral']:
                add('staffing',f'L{demand:g}_d{d2:g}_{variant}',Model(demand=demand,premium=15,d2=d2,
                    gamma=0. if variant=='one_referral' else -1.,servers=18,nu=54.))
    return tasks

def solve_model(m,regime='dual',seed=41,required_slack=1e-3):
    if is_regional_baseline(m):
        return optimize_regional(m,regime,scalar_maxfun=900,inner_maxfun=120,
                                 outer_maxfun=220,required_slack=required_slack)
    return optimize(m,regime,seed=seed,iterations=450,restarts=6,required_slack=required_slack)

def validated_referral(task):
    path=OUT/'regional_factorial_validation.json'
    if task['group']!='referral' or not path.exists():return None
    payload=json.loads(path.read_text())
    row=next((row for row in payload['rows'] if row['label']==task['label'] and row['regime']==task['regime']),None)
    if row is None:return None
    m=Model(**task['model']);r0=row['regional']
    result=evaluate(m,(r0['mu_on'],r0['p_on'],r0['p_off']))
    if result is None:raise ValueError('Validated regional policy failed reconstruction')
    result['solver']='regional-validated';result['regional_support']=r0['region']
    result['search_best']=result['objective'];result['search_spread']=0.
    return result

def run_task(t):
    signature={**t,'solver_version':'regional-v1'}
    key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:16]
    path=OUT/'cache'/f'{key}.json'
    if path.exists():return json.loads(path.read_text())
    m=Model(**t['model'])
    start=time.time()
    r=validated_referral(t) or solve_model(m,t['regime'],seed=41)
    result={**t,'result':r,'seconds':time.time()-start,'key':key}
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result,indent=2))
    return result

def run(workers):
    tasks=design(); (OUT/'design.json').write_text(json.dumps(tasks,indent=2))
    results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(run_task,t):t for t in tasks}
        for i,f in enumerate(as_completed(futures),1):
            result=f.result();results.append(result)
            if i%10==0 or result['group']=='staffing':print(f'{i}/{len(tasks)} {result["group"]} {result["label"]} J={result["result"]["objective"]:.3f}',flush=True)
    results.sort(key=lambda t:(t['group'],t['label'],t['regime']))
    (OUT/'experiments.json').write_text(json.dumps(results,indent=2))
    transfers=[]
    # Same environment, different model used to select physical capacity and prices.
    for row in results:
        if row['group']!='referral' or row['model']['d2']==0:continue
        m=Model(**row['model'])
        base=next(t for t in results if t['group']=='referral' and t['model']['d2']==0
            and t['model']['demand']==m.demand and t['model']['premium']==m.premium and t['regime']==row['regime'])
        r=evaluate(m,policy(base['result']))
        transfers.append(dict(label=row['label'],regime=row['regime'],model=row['model'],
            full=row['result'],simple=base['result'],transferred=r,
            loss=None if r is None else row['result']['objective']-r['objective']))
    (OUT/'transfers.json').write_text(json.dumps(transfers,indent=2))
    # Map continuous capacity to the nearest integer clinician count in the same budget.
    staffing=[]
    for row in results:
        if row['group']!='staffing' or row['model']['gamma']==0:continue
        m=Model(**row['model'])
        continuous=solve_model(replace(m,servers=0),seed=41)
        k=int(np.clip(np.floor(continuous['mu_on']/m.nu+.5),1,m.servers-1))
        transferred=evaluate(m,(k,continuous['p_on'],continuous['p_off']))
        staffing.append(dict(label=row['label'],full=row['result'],continuous=continuous,
            transferred=transferred,model=row['model'],
            loss=None if transferred is None else row['result']['objective']-transferred['objective']))
    (OUT/'staffing_transfers.json').write_text(json.dumps(staffing,indent=2))
    print('Completed',len(results),'registered configurations',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4)
    run(p.parse_args().workers)
