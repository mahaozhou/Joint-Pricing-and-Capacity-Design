"""High-precision checks of retained equilibrium policies and global uppers."""
from dataclasses import fields
from collections import Counter
from pathlib import Path
import random
import json
import math

import mpmath as mp
import pandas as pd

from model import Scenario, evaluate
from experiments import RESULTS


def exact_equilibrium(c,k,p):
    mp.mp.dps=42
    M=lambda x: mp.mpf(str(x))
    a,b,d,th,alpha=map(M,(c.a,c.b,c.d,c.theta,c.alpha))
    if c.path=='N':
        b=M(0)
    gamma=M(c.gamma if c.path=='R' else 0)
    z,ze=1-gamma*a*b,1-gamma*a*d
    A=1 if c.path in ('N','O') else 1+a*b/z
    q=[(A,a/z),(b/z,1/z)]
    h=[(1+th*(A-1),a/z),(th*b/z,1/z)]
    v=[M(c.V)+a*M(c.s)-M(c.T)*a/z,M(c.V)+M(c.s)-M(c.T)/z]
    e=(d/ze,1/ze); be=(th*d/ze,1/ze)
    p=list(map(M,p)); rates=list(map(M,(c.nu_v,c.nu_f)))
    delay=list(map(M,(c.cv,c.cf))); cap=(k,c.K-k)
    E=M(c.required)*mp.exp(-M(c.elasticity)*max(M(0),alpha*(sum(x*y for x,y in zip(p,be))-M(c.fixed_f))))
    lam=M(c.demand)
    ext=[E*ee for ee in e]
    wd=[lambda z,j=j: exact_delay(z,cap[j],rates[j]) for j in range(2)]
    maxload=[cap[j]*rates[j] for j in range(2)]
    B=[sum(p[j]*h[i][j] for j in range(2)) for i in range(2)]
    net=[v[i]-alpha*B[i] for i in range(2)]

    def times(x):
        return [wd[j](ext[j]+sum(q[i][j]*x[i] for i in range(2))) for j in range(2)]

    def util(x):
        w=times(x)
        return [net[i]-sum(q[i][j]*delay[j]*w[j] for j in range(2)) for i in range(2)]

    def stable(x):
        return all(ext[j]+sum(q[i][j]*x[i] for i in range(2))<maxload[j] for j in range(2))

    def domain(origin,direction):
        l,h=M(0),lam
        for j in range(2):
            kappa=sum(q[i][j]*direction[i] for i in range(2))
            spare=maxload[j]-ext[j]-sum(q[i][j]*origin[i] for i in range(2))
            if kappa>0:h=min(h,spare/kappa)
            if kappa<0:l=max(l,spare/kappa)
            if kappa==0 and spare<=0:return (M(1),M(0))
        return l,h

    options=[(M(0),M(0)),(lam,M(0)),(M(0),lam)]
    for i in range(2):
        vec=tuple(M(int(j==i)) for j in range(2))
        l,h0=domain((0,0),vec)
        if h0>=l:
            def at(t):
                x=[z*t for z in vec]
                if not stable(x):return -mp.inf
                return util(x)[i]
            if at(l)>=0>=at(h0):
                root=bisect(l,h0,at)
                options.append(tuple(w*root for w in vec))
    l,h0=domain((0,lam),(1,-1))
    if h0>l:
        shrink=mp.mpf("1e-25")
        lo,hi=l+shrink,h0-shrink
        def f(t):
            u=util((t,lam-t))
            return u[0]-u[1]
        if hi>=lo and f(lo)>=0>=f(hi):
            t=bisect(lo,hi,f)
            options.append((t,lam-t))

    # Partial participation on both routes follows from zero utilities.
    det=q[0][0]*q[1][1]-q[0][1]*q[1][0]
    desired=((q[1][1]*net[0]-q[0][1]*net[1])/det/delay[0],
             (q[0][0]*net[1]-q[1][0]*net[0])/det/delay[1])
    if all(desired[j]>=wd[j](ext[j]) for j in range(2)):
        l=[bisect(ext[j],maxload[j],lambda l,j=j:desired[j]-wd[j](l))-ext[j] for j in range(2)]
        options.append(((q[1][1]*l[0]-q[1][0]*l[1])/det,
                        (q[0][0]*l[1]-q[0][1]*l[0])/det))
    for x in options:
        if min(x)<0 or sum(x)>lam or not stable(x):
            continue
        u=util(x);top=max(*u,0)
        res=max([top-u[i] for i in range(2) if x[i]>0]+[top if sum(x)<lam else 0])
        if res > mp.mpf("1e-25"):
            continue
        G=sum(B[i]*x[i] for i in range(2))+M(c.receipt)*E*sum(p[j]*be[j] for j in range(2))
        return tuple(float(l) for l in x),float(G-M(c.penalty)*(lam-sum(x)))
    raise RuntimeError((c,k,p))


def bisect(lo,hi,f):
    for _ in range(116):
        mid=(lo+hi)/2
        if f(mid)>0:lo=mid
        else:hi=mid
    return (lo+hi)/2


def exact_delay(load,k,rate):
    if load>=k*rate:return mp.inf
    z=load/rate
    t=mp.mpf(1)
    total=t
    for i in range(1,k):
        t*=z/i
        total+=t
    B=t*z*rate/(k*rate-load)
    return 1/rate+B/(total+B)/(k*rate-load)


def main():
    t=pd.read_csv(RESULTS/'experiments.csv')
    rng=random.Random(20261124)
    unique=t.drop_duplicates('scenario_id').reset_index(drop=True)
    names={f.name for f in fields(Scenario)}
    selected=[]
    for _,g in unique.groupby('path'):
        selected+=rng.sample(list(g.index),min(15,len(g)))
    for _,g in unique.groupby('support'):
        selected+=rng.sample(list(g.index),min(3,len(g)))
    errors=[]; supports=Counter()
    for i in sorted(set(selected)):
        row=unique.iloc[i]
        c=Scenario(**{n:row[n] for n in names})
        k=int(row.k_v)
        x,j=exact_equilibrium(c,k,[row.p_v,row.p_f])
        err=max(abs(x[0]-row.x_v),abs(x[1]-row.x_f))
        assert row.lower_bound-2e-7<=j<=row.point_upper+2e-7
        assert err<3e-6,(c,err)
        errors.append(dict(scenario_id=row.scenario_id,flow_abs_error=err,
                           objective_abs_error=abs(j-row.J),support=row.support))
        supports[row.support]+=1
    stresses=0
    for i in rng.sample(list(unique.index),250):
        row=unique.iloc[i]
        c=Scenario(**{n:row[n] for n in names})
        kv=(int(row.fixed_k) if pd.notna(row.fixed_k) and row.fixed_k else rng.randrange(1,int(c.K)))
        pv_a,pv_b,pf_a,pf_b=c.box
        p,q=rng.uniform(pv_a,pv_b),rng.uniform(pf_a,pf_b)
        evaluated=evaluate(c,kv,p,q)
        if evaluated is not None:
            assert evaluated['lower_bound']<=row.upper_bound+1e-6,(row.scenario_id,kv,p,q)
            stresses+=1
    pd.DataFrame(errors).to_csv(RESULTS/'independent_numeric_audit.csv',index=False)
    result=dict(mpmath_decimal_digits=42,retained_policies=len(errors),
                supports=dict(supports),stress_policies=stresses,
                largest_flow_abs_error=max(v['flow_abs_error'] for v in errors),
                largest_objective_abs_error=max(v['objective_abs_error'] for v in errors))
    (RESULTS/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
