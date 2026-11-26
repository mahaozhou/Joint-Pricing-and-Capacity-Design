"""Replicated stationary DES at prespecified-support selected policies."""
from concurrent.futures import ProcessPoolExecutor
from ctypes import CDLL, c_double, c_uint64, POINTER
from dataclasses import fields
from pathlib import Path
import argparse
import json
import math
import time

import numpy as np
import pandas as pd
from scipy.stats import t as student

from model import Scenario, build_shared, shared_library
RESULTS = Path(__file__).resolve().parents[1] / "results"


HERE = Path(__file__).resolve().parent
CLASSES = ['virtual-first','in-person-first','mandatory']
STAGES = ['virtual-initial','in-person-initial','in-person-referred','virtual-follow-up']
FIELDS = {f.name for f in fields(Scenario)}


def get_cases(data):
    # Deterministic priority for six distinct supports/contracts, plus high demand.
    # Structural witnesses use exact selected endpoints. Other classes use
    # certified policies, lowest utilization first, then design row order.
    path_set = data[(data.experiment=='paths') & (data.path=='S') &
                    (data.status=='epsilon_certified')]
    exact = data[data.experiment=='policy_switch']
    loop = data[(data.experiment=='recurrent') & (data.gamma==1) &
                (data.status=='epsilon_certified')]
    priorities = {
        'virtual_only':exact[exact.b==.245],
        'offline_only':exact[exact.b==.255],
        'mixed':path_set[path_set.entry=='both'],
        'partial_exit':path_set[(path_set.access>1e-4)&(path_set.access<.9999)],
        'free_followup':path_set[(path_set.theta==0)&(path_set.b>0)&(path_set.access>.01)],
        'recurrent':loop[loop.access>.01],
        'high_load':path_set[(path_set.demand==878)&(path_set.access>.01)],
    }
    used = set()
    output={}
    for label,t in priorities.items():
        t=t.assign(peak=t[['utilization_v','utilization_f']].max(axis=1))
        t=t.sort_values('peak',ascending=(label!='high_load'),kind='stable')
        t=t[~t.scenario_id.isin(used)]
        if t.empty:
            raise ValueError(f"No distinct usable DES case for {label}")
        row=t.iloc[0].to_dict()
        output[label]=row
        used.add(row['scenario_id'])
    return output


def run(task):
    group,row,rep,warm,obs=task
    sc=Scenario(**{k:row[k] for k in FIELDS})
    seed=20261124+1000*list(NAMES).index(group)+rep+(100000 if warm>50 else 0)
    dll=CDLL(str(shared_library("simulate")))
    ptr=POINTER(c_double)
    dll.simulate.argtypes=ptr,c_uint64,ptr
    dll.simulate.restype=__import__('ctypes').c_int
    data=np.array([sc.demand,row['x_v'],row['x_f'],row['E'],sc.a,sc.b,sc.d,sc.gamma,
                   'NOSR'.index(sc.path),sc.theta,row['p_v'],row['p_f'],sc.nu_v,sc.nu_f,
                   row['k_v'],row['k_f'],warm,obs,sc.penalty,sc.receipt],float)
    out=np.zeros(56)
    start=time.perf_counter()
    rc=dll.simulate(data.ctypes.data_as(ptr),seed,out.ctypes.data_as(ptr))
    if rc!=1:
        raise RuntimeError((group,rep,rc))
    status=dict(case=group,rep=rep,seed=seed,warmup=warm,measurement=obs,
                scenario_id=row['scenario_id'],seconds=time.perf_counter()-start)
    for i,j in enumerate('vf'):
        n,w,q=out[3*i:3*i+3]
        status.update({f'load_{j}':n/obs,f'W_{j}':w/n if n else np.nan,
                       f'Wq_{j}':q/n if n else np.nan})
    Q,H,e,he,val=sc.visits
    cost=np.array([sc.cv*status['W_v'],sc.cf*status['W_f']])
    # Passive probes use pooled delays. No probe creates visits.
    probe=val-sc.alpha*np.array([row['p_v'],row['p_f']])@H-cost@Q
    status.update(U_v_probe=probe[0],U_f_probe=probe[1],
                  U_difference_probe=probe[0]-probe[1])
    stages=[]
    for i,route in enumerate(CLASSES):
        for j,stage in enumerate(STAGES):
            ix=6+3*(i*4+j)
            n,w,q=out[ix:ix+3]
            if n:
                stages.append(dict(**status_stub(status),route=route,stage=stage,n=n,
                                   load=n/obs,W=w/n,Wq=q/n))
    g=0
    for i,label in enumerate(('v','f','E')):
        n,online,offline,bills=out[42+4*i:46+4*i]
        status[f'x_{label}']=n/obs
        if n:
            status[f'D_{label}']=(online+offline)/n
            status[f'B_{label}']=bills/n
            if i<2:
                status[f'U_{label}_direct']=val[i]-(sc.alpha*bills+sc.cv*online+sc.cf*offline)/n
        g+=bills
    status['G']=g/obs
    status['J']=(g-sc.penalty*out[54]+(sc.receipt-1)*out[53])/obs
    status['balks']=out[54]/obs
    status['events']=int(out[55])
    return status,stages


def status_stub(s):
    return {k:s[k] for k in ('case','rep','seed','scenario_id')}


NAMES=('virtual_only','offline_only','mixed','partial_exit','free_followup','recurrent','high_load')


def summarize(df,groups,metrics):
    ans=[]
    for key, sub in df.groupby(groups,dropna=False,sort=False):
        if not isinstance(key,tuple):
            key=(key,)
        prefix=dict(zip(groups,key))
        for field in metrics:
            values=sub[field].dropna()
            if not len(values):continue
            half=student.ppf(.975,len(values)-1)*values.std(ddof=1)/math.sqrt(len(values))
            mean=values.mean()
            ans.append(dict(**prefix,metric=field,n=len(values),mean=mean,
                            lower_95=mean-half,upper_95=mean+half))
    return pd.DataFrame(ans)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--replications',type=int,default=100)
    ap.add_argument('--warmup',type=int,default=50)
    ap.add_argument('--measurement',type=int,default=500)
    ap.add_argument('--high-load-only',action='store_true')
    args=ap.parse_args()
    build_shared("simulate")
    cases=get_cases(pd.read_csv(RESULTS/'experiments.csv'))
    if args.high_load_only:
        cases={'high_load':cases['high_load']}
    prefix='des_extended' if args.high_load_only else 'des'
    print('DES cases:',[(k,v['scenario_id'],v['entry']) for k,v in cases.items()],flush=True)
    items=[(k,v,i,args.warmup,args.measurement) for k,v in cases.items() for i in range(args.replications)]
    raw=[]; stage_raw=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,(r,st) in enumerate(pool.map(run,items,chunksize=1),1):
            raw.append(r);stage_raw+=st
            if i%30==0:print(f'{i}/{len(items)} DES replications complete',flush=True)
    reps=pd.DataFrame(raw); stage=pd.DataFrame(stage_raw)
    reps.to_csv(RESULTS/f'{prefix}_replications.csv',index=False)
    stage.to_csv(RESULTS/f'{prefix}_stages_raw.csv',index=False)
    summary=summarize(reps,['case'],[k for k in reps.columns if k not in (
        'case','rep','seed','warmup','measurement','scenario_id','events')])
    analytical={}
    for name,row in cases.items():
        sc=Scenario(**{k:row[k] for k in FIELDS})
        Q,H,e,he,val=sc.visits
        W=np.array([row['W_v'],row['W_f']])
        D=W@Q
        p=np.array([row['p_v'],row['p_f']])
        analytical[name]=dict(W_v=W[0],W_f=W[1],Wq_v=W[0]-1/sc.nu_v,Wq_f=W[1]-1/sc.nu_f,
             load_v=row['ell_v'],load_f=row['ell_f'],x_v=row['x_v'],x_f=row['x_f'],x_E=row['E'],
             U_v_probe=row['U_v'],U_f_probe=row['U_f'],U_difference_probe=row['U_v']-row['U_f'],
             U_v_direct=row['U_v'],U_f_direct=row['U_f'],D_v=D[0],D_f=D[1],D_E=W@e,
             B_v=(p@H)[0],B_f=(p@H)[1],B_E=p@he,G=row['G'],J=row['J'],
             balks=sc.demand-row['x_v']-row['x_f'])
    summary['analytic']=[analytical[g].get(m,np.nan) for g,m in zip(summary.case,summary.metric)]
    summary['inside_95']=(summary.lower_95<=summary.analytic)&(summary.analytic<=summary.upper_95)
    summary.to_csv(RESULTS/f'{prefix}_summary.csv',index=False)
    stage_summary=summarize(stage,['case','route','stage'],['W','Wq','load'])
    expected=[]
    for _,z in stage_summary.iterrows():
        pool='v' if z.stage.startswith('virtual') else 'f'
        expected.append(analytical[z['case']].get(f'{z.metric}_{pool}',np.nan) if z.metric!='load' else np.nan)
    stage_summary['pool_analytic']=expected
    stage_summary['inside_95']=(stage_summary.lower_95<=stage_summary.pool_analytic)&(stage_summary.pool_analytic<=stage_summary.upper_95)
    stage_summary.to_csv(RESULTS/f'{prefix}_stage_summary.csv',index=False)
    pd.DataFrame.from_dict(cases,orient='index').rename_axis('case').to_csv(RESULTS/f'{prefix}_policies.csv')
    (RESULTS/f'{prefix}_design.json').write_text(json.dumps(vars(args),indent=2)+'\n')


if __name__=='__main__':
    main()
