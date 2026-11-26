"""V2 scenario registry, budgeted global runs, and matched policy transfers."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from itertools import product
import hashlib
import json
import math
import os
from pathlib import Path
import time

import pandas as pd
from model import Scenario, build, evaluate, optimum, full_flow
from verify_selected import scale_witness


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BREF = .2
TARGET = .1


def canonical(c):
    # Identical fixed-price domains need not repeat based on inactive cap fields.
    if c.regime == 'F0':
        c = replace(c,cap_v=c.fixed_v,cap_f=c.fixed_f)
    if c.regime == 'F1':
        c = replace(c,fixed_v=0,cap_f=c.fixed_f)
    if c.elasticity == 0 and c.regime == 'F2':
        c = replace(c,fixed_v=0,fixed_f=40)
    if c.path == 'N':
        c = replace(c,path='S',b=0)
    if c.path == 'O' and c.a*c.b == 0:
        c = replace(c,path='S')
    if c.path == 'R' and c.gamma == 0:
        c = replace(c,path='S')
    if c.required == 0:
        c = replace(c,receipt=0,elasticity=0,d=0)
    if c.required*c.d == 0 and c.b == 0:
        c = replace(c,theta=0)
    return c


def make_id(sc, fixed=0):
    z = canonical(sc).encode()
    return hashlib.sha256(z.tobytes()+fixed.to_bytes(2,'little')).hexdigest()[:16]


def registry():
    rows = []
    def add(group, c, **extra):
        rows.append((group,c,extra))
    for L,s,b,theta,reg in product((572,878),(15,30),(0,.2,.4,.6,.8),(0,.5,1),('F0','F1','F2')):
        for path in 'NOS':
            add('paths',Scenario(demand=L,s=s,b=b,theta=theta,regime=reg,path=path))
    for L,s,theta,reg in product((572,878),(15,30),(0,.5,1),('F1','F2')):
        for i in range(41):
            add('followup_scan',Scenario(demand=L,s=s,b=i/50,theta=theta,regime=reg))
    for L,b,a,theta,reg in product((572,878),(.1,.35,.65),(.1,.2,.35,.5,.65,.8),(0,.5,1),('F0','F1','F2')):
        add('independent_ab',Scenario(demand=L,a=a,b=b,theta=theta,regime=reg))
    for L,b,reg in product((572,878),(.2,.6),('F1','F2')):
        c=Scenario(demand=L,b=b,regime=reg)
        for name,values in [('d',(0,.25,.5,.75)),('required',(0,30,60,90)),
                            ('alpha',(.3,.6,1)),('nu_v',(48,54,60))]:
            for v in values:
                add('single_factor',replace(c,**{name:v}),varied=name)
    for L,b,theta,reg in product((572,878),(.2,.6),(0,.5,1),('F0','F1','F2')):
        c=Scenario(demand=L,b=b,theta=theta,regime=reg)
        add('mandatory_accounts',replace(c,required=0,receipt=0),account='C0')
        add('mandatory_accounts',replace(c,receipt=0),account='C1')
        add('mandatory_accounts',c,account='C2')
    for L,b,theta,alpha,eta in product((572,878),(.2,.6),(.5,1),(.6,1),(0,.01,.03)):
        add('mandatory_elasticity',Scenario(demand=L,b=b,theta=theta,alpha=alpha,elasticity=eta))
    for L,b,theta,reg,gamma in product((572,878),(.1,.35,.65),(0,.5,1),('F1','F2'),(0,.5,1)):
        add('recurrent',Scenario(demand=L,b=b,theta=theta,regime=reg,path='R',gamma=gamma))
    cap=float(scale_witness()['cap'][0])
    p=Scenario(alpha=.3,cap_v=cap,regime='F1')
    for b,gamma in product(sorted(set([i/100 for i in range(20,36)]+[.245,.2475,.2525,.255])),(0,.5,1)):
        add('recurrent_switch',replace(p,b=b,path='R',gamma=gamma))
    for i in range(20,36):
        add('policy_switch',replace(p,b=i/100))
    for b in (.245,.2475,.2525,.255):
        add('policy_switch',replace(p,b=b))
    add('policy_switch',replace(p,b=.25),witness='tie')
    for L,theta,reg in product((572,878),(0,.5,1),('F0','F1','F2')):
        add('zero_mandatory_single_direction',Scenario(demand=L,b=0,required=0,theta=theta,regime=reg))
    for path in 'OS':
        add('exact_small_role',Scenario(demand=.2,required=.05,a=.5,b=.75,d=.2,theta=0,s=17.4,
                        V=16.8,cv=4,cf=1,penalty=1,nu_v=1,nu_f=1,K=5,
                        cap_v=5.25,cap_f=10,path=path))
        add('exact_small_gain',Scenario(demand=2.3,required=.3,a=1/3,b=.4,d=.3,theta=0,s=24,
                        V=8,cv=1,cf=3,penalty=2,nu_v=1,nu_f=1,K=6,
                        cap_v=3,cap_f=6,path=path))
    return rows


def run_one(task):
    name,c,fixed,budget,tol=task
    try:
        best,splits=optimum(c,fixed=fixed,budget=budget,target=tol)
        return dict(id=name, fixed_k=fixed, parameters=asdict(c), best=best, splits=splits)
    except Exception as e:
        return dict(id=name, fixed_k=fixed, parameters=asdict(c), error=repr(e),budget=budget)


def safe_obj(o):
    if isinstance(o,float) and not math.isfinite(o):
        return None
    if isinstance(o,list):
        return [safe_obj(v) for v in o]
    if isinstance(o,dict):
        return {k:safe_obj(v) for k,v in o.items()}
    return o


def load_cache(path):
    done={}
    if path.exists():
        for line in path.read_text().splitlines():
            result=json.loads(line)
            old=done.get(result['id'])
            if old is None or 'error' in old or ('best' in result and
                     result['best']['gap']<old['best']['gap']):
                done[result['id']]=result
    return done


def dispatch(rows,done,cache,workers,budget,tol,retry_gap=None):
    queue={}
    for group,c,extra in rows:
        fixed=extra.get('fixed_k',0)
        key=make_id(c,fixed)
        old=done.get(key)
        retry=old is None or 'error' in old
        if not retry and retry_gap is not None:
            retry=old['best']['gap']>retry_gap and old['best']['node_limit']<budget
        if retry:
            queue[key]=(key,canonical(c),fixed,budget,tol)
    if not queue:
        return
    print(f"Running {len(queue)} unique searches, budget {budget}",flush=True)
    count=Counter()
    t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool, cache.open('a') as f:
        for ix,v in enumerate(pool.map(run_one,queue.values(),chunksize=3),1):
            old=done.get(v['id'])
            if old is None or 'error' in old or ('best' in v and v['best']['gap']<old['best']['gap']):
                done[v['id']]=v
            count[v.get('best',{}).get('status','error')]+=1
            f.write(json.dumps(safe_obj(v),allow_nan=False)+'\n')
            if ix%25==0 or ix==len(queue):
                f.flush()
                print(f"{ix}/{len(queue)} in {time.perf_counter()-t:.1f}s {dict(count)}",flush=True)


def coarsened_role(row):
    return row['k_v'],row['entry']


def keys(*args):
    return args


def records(rows,done):
    tbl=[]
    for group,c,extra in rows:
        id=make_id(c,extra.get('fixed_k',0)); base=done[id]
        rec=dict(experiment=group,scenario_id=id,**asdict(c),**extra)
        rec.update(price_lo_v=c.box[0],price_hi_v=c.box[1],price_lo_f=c.box[2],price_hi_f=c.box[3])
        if 'error' in base:
            rec.update(status='error',error=base['error'])
        else:
            rec.update(base['best'])
            rec['possible_optimal_k']='|'.join(map(str,base['best']['possible_optimal_k']))
            rec['clinician_time_difference_minutes']=480*(np_dot_times(c))
            rec['residual_capacity_v']=rec['k_v']*c.nu_v-rec['E']*c.visits[2][0]
            rec['residual_capacity_f']=rec['k_f']*c.nu_f-rec['E']*c.visits[2][1]
            rec['utilization_v']=rec['ell_v']/(rec['k_v']*c.nu_v)
            rec['utilization_f']=rec['ell_f']/(rec['k_f']*c.nu_f)
        tbl.append(rec)
    return pd.DataFrame(tbl)


def np_dot_times(c):
    q=c.visits[0]
    return (q[0,0]-q[0,1])/c.nu_v+(q[1,0]-q[1,1])/c.nu_f


def refinements(rows,done):
    ref=[]
    curves={}
    for group,c,extra in rows:
        if group=='followup_scan':
            curves.setdefault((c.demand,c.s,c.theta,c.regime),[]).append(c)
    for ls in curves.values():
        for a,b in zip(ls,ls[1:]):
            av=done[make_id(a)].get('best')
            bv=done[make_id(b)].get('best')
            if av and bv and coarsened_role(av)!=coarsened_role(bv):
                for i in (1,2,3):
                    ref.append(('followup_refined',replace(a,b=round(a.b+i*(b.b-a.b)/4,5)),{}))
    return ref


def fixed_staff(rows,done):
    ret=[]
    for group,c,extra in rows:
        if group in ('policy_switch','recurrent_switch') and c.gamma==0:
            ret.append(('price_only_policy',c,{'fixed_k':13}))
        if group.startswith('followup_'):
            base=done[make_id(replace(c,b=BREF))].get('best')
            if base:
                ret.append(('price_only_reference',c,{'fixed_k':base['k_v'],'reference_b':BREF}))
    return ret


def save_tables(rows,done):
    tbl=records(rows,done)
    tbl.to_csv(RESULTS/'experiments.csv',index=False)
    s=[]
    for k,value in done.items():
        for p in value.get('splits',[]):
            row=dict(scenario_id=k,**value['parameters'],fixed_k=value['fixed_k'],**p)
            s.append(row)
    pd.DataFrame(s).to_csv(RESULTS/'staff_bounds.csv',index=False)
    return tbl


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--workers',type=int,default=7)
    cli.add_argument('--budget',type=int,default=2000)
    cli.add_argument('--retry',type=int,default=0)
    cli.add_argument('--no-build',action='store_true')
    cli.add_argument('--base-only',action='store_true')
    cli.add_argument('--tight-scenarios',action='store_true')
    args=cli.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if not args.no_build:
        build()
    cache=RESULTS/'raw_searches.jsonl'
    done=load_cache(cache)
    rows=registry()
    dispatch(rows,done,cache,args.workers,args.budget,TARGET)
    if args.base_only:
        save_tables(rows,done)
        return
    if args.retry:
        relax=[row for row in rows if row[0] in ('paths','policy_switch','exact_small_role','exact_small_gain')]
        if args.tight_scenarios:
            relax=rows
        dispatch(relax,done,cache,args.workers,args.retry,TARGET,retry_gap=TARGET)
    rows += refinements(rows,done)
    dispatch(rows,done,cache,args.workers,args.budget,TARGET)
    rows += fixed_staff(rows,done)
    dispatch(rows,done,cache,args.workers,args.budget,TARGET)
    if args.retry:
        levels = rows if args.tight_scenarios else [row for row in rows if row[0] == 'price_only_policy']
        dispatch(levels,done,cache,args.workers,args.retry,TARGET,retry_gap=TARGET)
    tbl=save_tables(rows,done)
    print("Output",len(tbl),"records",tbl['status'].value_counts().to_dict(),flush=True)


if __name__=='__main__':
    main()
