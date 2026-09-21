"""Common operational reserves in full and required-load-aware planning models."""
from dataclasses import replace,asdict
from concurrent.futures import ProcessPoolExecutor
import json
from model import Model,evaluate,policy
from experiments import OUT,solve_model

def one(args):
    L,d,slack=args
    m=Model(demand=L,premium=15.,d2=d)
    simple=replace(m,d2=0.,gamma=0.,dE=d)
    full=solve_model(m,seed=1409,required_slack=slack)
    selected=solve_model(simple,seed=1709,required_slack=slack)
    transferred=evaluate(m,policy(selected))
    assert transferred is not None
    loss=full['objective']-transferred['objective']
    assert loss>-.01
    return dict(model=asdict(m),required_slack=slack,full=full,selected=selected,transferred=transferred,loss=loss)

if __name__=='__main__':
    cases=[(L,d,s) for s in [54.,108.] for L in [572.,878.] for d in [.1,.4,.7]]
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,cases))
    (OUT/'reserve_sensitivity.json').write_text(json.dumps(rows,indent=2))
    print([(r['required_slack'],r['model']['demand'],r['model']['d2'],round(r['loss'],2)) for r in rows],flush=True)
