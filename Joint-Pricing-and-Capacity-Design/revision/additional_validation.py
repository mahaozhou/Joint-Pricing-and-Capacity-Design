from pathlib import Path
from dataclasses import asdict
import json,time,itertools
import numpy as np
from scipy.optimize import linprog
from numba import set_num_threads
from model import Model,evaluate,policy
from simulation import simulate,OUT
from simulation import cache_matches

def save(name,m,r,mode=0,warm=10.,duration=20.,seed=1910000):
    file=OUT/f'des_{name}.json'
    if file.exists():
        cached=json.loads(file.read_text())
        if cache_matches(cached,m,r,mode=mode,seed=seed,warm=warm,duration=duration):return cached
    start=time.time();vals,s=simulate(m,r,mode=mode,warm=warm,duration=duration,seed=seed)
    s.update(name=name,seconds=time.time()-start)
    np.savez_compressed(OUT/f'des_{name}_replications.npz',values=vals)
    file.write_text(json.dumps(s,indent=2))
    print(name,'10000 replications',round(s['seconds'],1),'seconds',flush=True)
    return s

def mixed_policy(m,flows,mu,surplus):
    q,e,b,be,v=m.matrices();flows=np.array(flows)
    loads=q@flows+e*m.required
    w=1/(np.array([mu,m.capacity-mu])-loads)
    prices=np.linalg.solve(m.copay*b.T,v-q.T@(np.array([m.cost_on,m.cost_off])*w)-surplus)
    r=evaluate(m,(mu,*prices))
    assert r is not None and np.max(np.abs(flows-[r['flow_on'],r['flow_off']]))<1e-4
    return r

def run():
    set_num_threads(4)
    m=Model(demand=572.,premium=15.,d2=.4)
    vf=mixed_policy(m,[180.,392.],430.,2.)
    a=save('prescribed_VF',m,vf)
    m2=Model(demand=878.,premium=15.,d2=.4)
    bvf=mixed_policy(m2,[220.,300.],500.,0.)
    b=save('prescribed_BVF',m2,bvf)
    rows=json.loads((OUT/'experiments.json').read_text())
    row=next(t for t in rows if t['group']=='staffing' and t['label']=='L878_d0.1_staged')
    c=save('staffing',Model(**row['model']),row['result'])
    d=save('long_window',m2,bvf,warm=20.,duration=40.)
    old=json.loads((OUT/'simulation.json').read_text())
    (OUT/'validation_all.json').write_text(json.dumps(old+[a,b,c,d],indent=2))
    base=json.loads((OUT/'des_moderate.json').read_text())
    mm=Model(**base['model']);r=base['policy']
    q,e,h,be,v=mm.matrices();w=np.array([r['wait_on'],r['wait_off']])
    # Same offline bundle, stationary flows, capacity, and receipts; largest offline tariff.
    bundle=mm.d2*r['p_on']+r['p_off']
    rhs=v[0]-(q.T@(np.array([mm.cost_on,mm.cost_off])*w))[0]
    lp=linprog([0,-1],A_ub=[-h[:,0]],b_ub=[-rhs],A_eq=[[mm.d2,1]],b_eq=[bundle],
        bounds=[(0,mm.cap_on),(0,mm.cap_off)],method='highs')
    assert lp.success
    alt=evaluate(mm,(r['decision'],*lp.x))
    assert alt and abs(alt['objective']-r['objective'])<.01
    screening=[]
    for i,(fc,fv,ff) in enumerate(itertools.product([.9,1.,1.1],[.8,1.,1.2],[.95,1.,1.05])):
        cand=evaluate(mm,(r['decision']*fc,min(60,r['p_on']*fv),min(80,r['p_off']*ff)))
        if cand:
            _,s=simulate(mm,cand,mode=1,reps=200,seed=3010000,warm=2,duration=4)
            screening.append(dict(policy=cand,objective=s['statistics']['objective']['mean']))
    selected=max(screening,key=lambda z:z['objective'])['policy']
    (OUT/'state_screening.json').write_text(json.dumps(screening,indent=2))
    state=[save('state_base',mm,r,mode=1,seed=4010000),
           save('state_max_offline',mm,alt,mode=1,seed=4010000),
           save('state_selected',mm,selected,mode=1,seed=4010000)]
    (OUT/'state_validation.json').write_text(json.dumps(state,indent=2))

if __name__=='__main__':run()
