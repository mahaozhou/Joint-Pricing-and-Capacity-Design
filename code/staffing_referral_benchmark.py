import json
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor,as_completed
from model import Model,evaluate,policy
from experiments import solve_model
from paths import STAFFING as OUT

def evaluate_one(row):
    m=Model(**row['model'])
    approximate=replace(m,d2=0.,gamma=0.,dE=m.d2)
    selected=solve_model(approximate,'dual',seed=731)
    transferred=evaluate(m,policy(selected))
    assert transferred is not None
    return dict(label=row['label'],model=row['model'],full=row['full'],selected=selected,
        transferred=transferred,loss=row['full']['objective']-transferred['objective'])

if __name__=='__main__':
    rows=json.loads((OUT/'staffing_transfers.json').read_text())
    with ProcessPoolExecutor(max_workers=4) as pool:ans=list(pool.map(evaluate_one,rows))
    assert min(r['loss'] for r in ans)>-.01
    (OUT/'staffing_referral_transfers.json').write_text(json.dumps(ans,indent=2))
    print([(r['label'],round(r['loss'],3)) for r in ans],flush=True)
