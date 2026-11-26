#!/usr/bin/env python3
"""Selected integer M/M/k witnesses. Standard library, exact rational arithmetic.
Derived from A verification/verify_structure.py and B Verification/verify.py.
These checks certify stated witnesses/upper bounds, not a full price-space solver.
"""
from dataclasses import dataclass, replace
from fractions import Fraction as F
from math import factorial
from pathlib import Path
import json

@dataclass(frozen=True)
class Case:
    a:F=F('.5'); b:F=F('.75'); d:F=F('.2'); theta:F=F(0)
    demand:F=F('.2'); required:F=F('.05'); V:F=F('16.8'); premium:F=F('-2.6')
    cost:tuple=(F(4),F(1)); rates:tuple=(F(1),F(1))
    alpha:F=F(1); r:F=F(1); cap:tuple=(F('5.25'),F(10)); K:int=5
    @property
    def n(self): return 1-self.a
    @property
    def values(self): return self.V+self.a*self.premium,self.V+self.premium

def wait(l,k,nu):
    z=l/nu
    if not 0<=z<k: raise ValueError('unstable pool')
    P=factorial(k-1)*sum((k-i)*z**i/factorial(i) for i in range(k))
    return 1/nu+z**k/(nu*(k-z)*P)

def bills(c,p,S=True):
    pv,pf=p; bf=pf+c.theta*c.b*pv
    return pv+c.a*(bf if S else pf),bf,pf+c.theta*c.d*pv

def policy(c,k,x,p=None,S=True):
    p=c.cap if p is None else p; xv,xf=map(F,x)
    assert 1<=k<c.K and all(0<=z<=cap for z,cap in zip(p,c.cap))
    assert min(xv,xf)>=0 and xv+xf<=c.demand
    qv=1+(c.a*c.b if S else 0)
    lv=c.required*c.d+qv*xv+c.b*xf; lf=c.required+c.a*xv+xf
    wv,wf=wait(lv,k,c.rates[0]),wait(lf,c.K-k,c.rates[1])
    bv,bf,be=bills(c,p,S); cv,cf=c.cost; vv,vf=c.values
    uv=vv-c.alpha*bv-qv*cv*wv-c.a*cf*wf
    uf=vf-c.alpha*bf-c.b*cv*wv-cf*wf
    G=xv*bv+xf*bf+c.required*be
    return dict(k=(k,c.K-k),x=(xv,xf),p=p,load=(lv,lf),W=(wv,wf),U=(uv,uf),G=G,J=G-c.r*(c.demand-xv-xf))

def wardrop(c,z,strict=False):
    x=z['x']; u=(*z['U'],F(0)); flows=(*x,c.demand-sum(x))
    if not all(y==0 or q==max(u) for y,q in zip(flows,u)): return False
    return not strict or all(q<max(u) for y,q in zip(flows,u) if y==0)

def hits(c,i,S=True,strict=False):
    x=(c.demand,0) if i==0 else (0,c.demand); out=[]
    for k in range(1,c.K):
        try: z=policy(c,k,x,S=S)
        except ValueError: continue
        if wardrop(c,z,strict): out.append(z)
    return out

def capval(c,S=True):
    bv,bf,be=bills(c,c.cap,S); return c.demand*max(bv,bf)+c.required*be

def role_change():
    out=[]
    for theta in (F(0),F('.02')):
        c=replace(Case(),theta=theta); L=F('.19'); bv,bf,be=bills(c,c.cap)
        D=(1+c.a*c.b)*c.cost[0]/c.rates[0]+c.a*c.cost[1]/c.rates[1]
        assert (max(bv,bf)+c.r)*L<=(bf+c.r)*c.demand
        assert min(bv,(c.values[0]-D)/c.alpha)<bf<bv
        s=policy(c,2,(0,c.demand)); o=policy(c,3,(c.demand,0),S=False)
        assert wardrop(c,s) and wardrop(c,o)
        assert o['J']==capval(c,False)>s['J']==c.demand*bf+c.required*be
        assert policy(c,4,(0,c.demand))['U'][1]<0
        at=policy(c,2,(0,c.demand),S=False)
        assert s['load']==at['load'] and s['G']==at['G'] and s['U'][1]==at['U'][1]
        assert at['U'][0]-s['U'][0]==c.a*c.b*(c.alpha*c.theta*c.cap[0]+c.cost[0]*s['W'][0])
        out.append(dict(theta=theta,S=s,O=o,screen_L=L,delay_floor=D))
    return out

def latent_gain():
    # B free-follow-up example: the same capped objective is attainable only in S.
    c=Case(a=F(1,3),b=F(2,5),d=F('.3'),theta=F(0),demand=F('2.3'),required=F('.3'),V=F(8),premium=F(4),cost=(F(1),F(3)),cap=(F(3),F(6)),K=6,r=F(2))
    bv,bf,be=bills(c,c.cap); assert bf>bv
    ss=hits(c,1); oo=hits(c,1,False)
    assert [z['k'][0] for z in ss]==[2] and not oo
    s=ss[0]; o=policy(c,2,(0,c.demand),S=False)
    assert s['J']==capval(c) and s['U'][0]<s['U'][1]<o['U'][0]
    return dict(S=s,O_at_same_target=o,O_cap_implementations=[],O_value_strictly_below=True)

def partial():
    c=Case(a=F('.4'),b=F('.5'),d=F('.2'),theta=F(1),demand=F('3.1'),required=F('.2'),V=F(3),premium=F(8),cost=(F(1),F(1)),cap=(F(25),F(2)),K=4)
    x=(0,F('1.5')); z=policy(c,2,x,(F(0),c.cap[1])); pv=z['U'][1]/(c.alpha*c.theta*c.b)
    z=policy(c,2,x,(pv,c.cap[1])); assert wardrop(c,z) and z['U'][0]<0
    pL=min(c.cap[0],(c.n*c.V-c.cost[0]/c.rates[0])/c.alpha)
    bv,bf,be=bills(c,(pL,c.cap[1])); UB=c.required*be+c.demand*max(bv,bf)
    exitUB=c.required*bills(c,c.cap)[2]-c.r*c.demand
    assert z['J']>max(UB,exitUB) and c.required+c.demand>=(c.K-1)*c.rates[1]
    return dict(witness=z,positive_virtual_upper=UB,exit_upper=exitUB,all_optima_partial_offline=True)

def switch():
    c=Case(a=F(1,4),b=F(1,3),d=F('.3'),theta=F('.5'),demand=F('3.1'),required=F('.3'),V=F(12),premium=F(0),cost=(F(2),F(2)),cap=(F(6),F(7)),K=7,r=F(2))
    lo,hi=replace(c,b=F('.32')),replace(c,b=F('.35'))
    assert c.b==(c.cap[0]-c.n*c.cap[1])/(c.n*c.theta*c.cap[0])
    for i,k in enumerate((5,2)):
        x=(c.demand,0) if i==0 else (0,c.demand)
        # Both route utilities at a fixed trial flow decrease in b. Endpoint
        # cross-test implies strict implementation throughout the interval.
        z0,z1=policy(lo,k,x),policy(hi,k,x)
        assert z1['U'][i]>max(F(0),z0['U'][1-i])
    low,high=hits(lo,0),hits(hi,1)
    assert [z['k'][0] for z in low]==[5] and [z['k'][0] for z in high]==[2]
    assert low[0]['J']==capval(lo) and high[0]['J']==capval(hi)
    assert 1+c.a+c.a*hi.b<1+hi.b
    v,f=policy(c,5,(c.demand,0)),policy(c,2,(0,c.demand))
    dv,df=(f['W'][j]-v['W'][j] for j in range(2))
    assert dv>0>df and df+F('.9')*dv>0
    assert F('.1')*F('.9')==c.required*c.d
    return dict(b0=c.b,low=low,high=high,W_change=(dv,df),cutoff=-df/dv,subgroup_change=(df,df+F('.9')*dv))

def scale_screen():
    # Explicitly separate designed policy scenarios from observed hospital calibration.
    # Set caps using a prespecified candidate b0; check all splits at b0.
    rows=[]
    for L in (572,878):
      for premium in (-5,10):
       for alpha in (F('.3'),F('.6'),F(1)):
        for theta in (F('.5'),F(1)):
         for b0 in (F('.25'),F('.5'),F('.75')):
          for pf in (F(40),):
            a=F('.35'); n=1-a; pv=n*pf/(1-n*theta*b0)
            c=Case(a=a,b=b0,d=F('.25'),theta=theta,demand=F(L),required=F(30),V=F(50),premium=F(premium),cost=(F(112),F(168)),rates=(F(54),F(54)),alpha=alpha,r=F(10),cap=(pv,pf),K=18)
            v,f=hits(c,0,strict=True),hits(c,1,strict=True)
            rows.append(dict(L=L,premium=premium,alpha=alpha,theta=theta,b0=b0,cap=c.cap,V_hits=[z['k'][0] for z in v],F_hits=[z['k'][0] for z in f],both=bool(v and f)))
    return rows

def scale_witness():
    # One passing case from the transparent 72-case screen; no claim of observed tariffs.
    a=F('.35'); b=F('.25'); theta=F('.5'); pv=(1-a)*40/(1-(1-a)*theta*b)
    c=Case(a=a,b=b,d=F('.25'),theta=theta,demand=F(572),required=F(30),V=F(50),premium=F(-5),cost=(F(112),F(168)),rates=(F(54),F(54)),alpha=F('.3'),r=F(10),cap=(pv,F(40)),K=18)
    lo,hi=replace(c,b=F('.245')),replace(c,b=F('.255'))
    margins=[]
    for i,k in enumerate((13,3)):
        x=(c.demand,0) if i==0 else (0,c.demand)
        z0,z1=policy(lo,k,x),policy(hi,k,x)
        margin=z1['U'][i]-max(F(0),z0['U'][1-i]); assert margin>0
        margins.append(margin)
    left,right=hits(lo,0),hits(hi,1)
    assert [z['k'][0] for z in left]==[13] and [z['k'][0] for z in right]==[3]
    assert left[0]['J']==capval(lo) and right[0]['J']==capval(hi)
    v,f=policy(c,13,(c.demand,0)),policy(c,3,(0,c.demand))
    dv,df=(f['W'][j]-v['W'][j] for j in range(2)); assert df<0<dv and df+dv>0
    return dict(b0=b,cap=c.cap,alpha=c.alpha,interval=(lo.b,hi.b),endpoint_cross_margins=margins,left=left,right=right,threshold_virtual=v,threshold_in_person=f,cutoff=-df/dv,W_change_minutes=(480*dv,480*df),subgroup_change_minutes=(480*df,480*(df+dv)))

def enc(z):
    if isinstance(z,F): return {'exact':str(z),'decimal':float(z)}
    raise TypeError(type(z))

if __name__=='__main__':
    out=dict(role_change=role_change(),latent_gain=latent_gain(),partial=partial(),switch=switch(),scale_screen=scale_screen(),scale_witness=scale_witness())
    file=Path(__file__).with_name('selected_results.json'); file.write_text(json.dumps(out,default=enc,indent=2),encoding='utf-8')
    rows=out['scale_screen']; print('PASS: exact role-change (free/paid), latent gain, partial-entry exclusion, paid switch and subgroup delay.')
    print('18-doctor cap-threshold feasibility screen:',sum(z['both'] for z in rows),'of',len(rows),'predefined policy scenarios pass both strict tests at b0.')
    print('PASS: 18-clinician designed policy witness, interval [0.245,0.255]; endpoint optimal counts 13 and 3.')
    print('No hospital-scale continuous-price scan or policy-transfer experiment is claimed by this script.')
