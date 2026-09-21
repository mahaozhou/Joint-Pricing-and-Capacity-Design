"""Isolates strategic referral misspecification from required-load accounting."""
import json
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor,as_completed
from model import Model,evaluate,policy
from experiments import OUT,solve_model

def run_one(row):
    m=Model(**row['model'])
    approximate=replace(m,d2=0.,gamma=0.,dE=m.d2 if m.dE<0 else m.dE)
    candidate=solve_model(approximate,row['regime'],seed=517)
    r=evaluate(m,policy(candidate))
    if r is None:raise RuntimeError('Required-aware transfer unexpectedly unstable')
    return dict(label=row['label'],regime=row['regime'],model=row['model'],full=row['full'],
                selected=candidate,transferred=r,loss=row['full']['objective']-r['objective'])

if __name__=='__main__':
    rows=json.loads((OUT/'transfers.json').read_text());ans=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,f in enumerate(as_completed([pool.submit(run_one,r) for r in rows]),1):
            ans.append(f.result())
            if i%12==0:print('required-aware transfers',i,'/',len(rows),flush=True)
    (OUT/'aware_transfers.json').write_text(json.dumps(ans,indent=2))
