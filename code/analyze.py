"""Matched contrasts, exact cap checks and analysis datasets for V2."""
from collections import Counter
from dataclasses import asdict, fields, replace
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from experiments import RESULTS, canonical, make_id
from model import Scenario, evaluate
import verify_selected as exact


KEYS=('demand','s','b','theta','regime')
PARAMS={g.name for g in fields(Scenario)}
UL='upper_bound'
LL='lower_bound'


def sc_of(v):
    return Scenario(**{key:v[key] for key in PARAMS})


def loss(dest,point):
    if point is None:
        return {'stable':False}
    lo=dest[LL]-point['point_upper']; hi=dest[UL]-point[LL]
    if hi< -1e-5:
        raise ArithmeticError(f'Feasible transfer exceeds destination upper bound. {hi}')
    return dict(stable=True,transfer_J=point['J'],transfer_G=point['G'],
                transfer_access=point['access'],transfer_mandatory_minutes=point['mandatory_minutes'],
                transfer_x_v=point['x_v'],transfer_x_f=point['x_f'],transfer_E=point['E'],
                transfer_ell_v=point['ell_v'],transfer_ell_f=point['ell_f'],
                transfer_W_v=point['W_v'],transfer_W_f=point['W_f'],
                transfer_point_lower=point[LL],transfer_point_upper=point['point_upper'],
                transfer_wardrop_residual=point['wardrop_residual'],
                loss_candidate=dest['J']-point['J'],loss_lower=max(0,lo),
                loss_upper=max(0,hi),gap_strict=lo>0,
                loss_normalized=(dest['J']-point['J'])/max(1.,abs(dest['J'])),
                delta_access=dest['access']-point['access'])


def transfer(dest,src,tag):
    row={name:dest[name] for name in ('demand','s','a','b','d','theta','alpha','regime','gamma','elasticity')}
    row.update(comparison=tag,source_id=src['scenario_id'],destination_id=dest['scenario_id'],
               origin_path=src['path'],destination_path=dest['path'],native_J_source=src['J'],
               native_J_destination=dest['J'],native_G_source=src['G'],native_G_destination=dest['G'],
               origin_status=src['status'],destination_status=dest['status'],
               origin_entry=src['entry'],destination_entry=dest['entry'],
               origin_k=src['k_v'],destination_k=dest['k_v'],
               native_access_source=src['access'],native_access_destination=dest['access'])
    point=evaluate(sc_of(dest),int(src['k_v']),src['p_v'],src['p_f'])
    row.update(loss(dest,point))
    row['entry_changed']=row['origin_entry']!=row['destination_entry']
    row['staff_changed']=row['origin_k']!=row['destination_k']
    return row


def path_analyses(t):
    data=t[t.experiment=='paths']
    pair=[]
    for key,z in data.groupby(list(KEYS),sort=False):
        p=z.set_index('path').to_dict('index')
        if key[2]==0:
            assert len(set(z.scenario_id))==1
        for src,dest in [('N','O'),('O','S'),('N','S')]:
            p[src]['path']=src
            p[dest]['path']=dest
            pair.append(transfer(p[dest],p[src],src+'->'+dest))
    s=pd.DataFrame(pair)
    assert s.stable.all()
    s.to_csv(RESULTS/'path_transfers.csv',index=False)
    trg=[]
    for keys,z in s.groupby(['regime','theta','comparison']):
        row=dict(zip(('regime','theta','comparison'),keys))
        q=z.loss_candidate.quantile([.25,.5,.75])
        row.update(count=len(z),strict_loss=int(z.gap_strict.sum()),
                   role_count=int(z.entry_changed.sum()),staff_count=int(z.staff_changed.sum()),
                   med=float(q[.5]),q1=float(q[.25]),q3=float(q[.75]),
                   med_rel=float(z.loss_normalized.median()),
                   med_access_pp=float(100*z.delta_access.median()),
                   max_abs_upper_gap=float((z.loss_upper-z.loss_lower).max()),
                   unresolved_origin_or_dest=int(((z.origin_status!='epsilon_certified')|(z.destination_status!='epsilon_certified')).sum()))
        trg.append(row)
    df=pd.DataFrame(trg);df.to_csv(RESULTS/'path_summary.csv',index=False)
    return s,df


def price_only(t):
    cols=['demand','s','theta','regime','b','alpha']
    v=t[t.experiment.isin(['followup_scan','followup_refined','policy_switch'])]
    f=t[t.experiment.isin(['price_only_reference','price_only_policy'])]
    f=f.drop_duplicates(cols)
    k=v.merge(f,on=cols,suffixes=('','_fixed')).drop_duplicates(cols)
    k['loss_lower']=(k[LL]-k[UL+'_fixed']).clip(lower=0)
    k['loss_upper']=(k[UL]-k[LL+'_fixed']).clip(lower=0)
    k['candidate_loss']=k.J-k.J_fixed
    if (k[UL]<k[LL+'_fixed']-1e-6).any():
        raise ArithmeticError("Fixed split exceeds joint upper bound.")
    k.to_csv(RESULTS/'repricing.csv',index=False)
    return k


def recurrent(t):
    a=[]
    for g in ('recurrent','recurrent_switch'):
        v=t[t.experiment==g]
        for _,s in v.groupby(list(KEYS)+['alpha'],sort=False):
            q=s.set_index('gamma').to_dict('index')
            same=q[0]
            same['gamma']=0
            for gamma,z in q.items():
                z['gamma']=gamma
                row=transfer(z,same,g)
                row.update(branch_value=z['J'],k_v=z['k_v'],route=z['entry'],
                           x_v=z['x_v'],x_f=z['x_f'],W_v=z['W_v'],W_f=z['W_f'],
                           ell_v=z['ell_v'],ell_f=z['ell_f'],
                           D_E=z['mandatory_minutes'],gap=z['gap'])
                a.append(row)
    df=pd.DataFrame(a);df.to_csv(RESULTS/'recurrent_transfers.csv',index=False)
    return df


def mandatory(t,staff):
    v=t[t.experiment=='mandatory_accounts'];m=[]
    for key,g in v.groupby(list(KEYS),sort=False):
        series=g.set_index('account').to_dict('index')
        c0,c1,c2=(series[x] for x in ('C0','C1','C2'))
        if c2['regime']=='F0':
            A=staff[staff.scenario_id==c1['scenario_id']].set_index('k_v')
            B=staff[staff.scenario_id==c2['scenario_id']].set_index('k_v')
            assert set(A.index)==set(B.index)
            bill=30*(c2['p_f']+c2['theta']*c2['d']*c2['p_v'])
            assert np.max(np.abs(B.J-A.J-bill))<2e-7
            assert np.max(np.abs(B.x_f-A.x_f))<2e-7
        for origin, destination in (('C0', 'C1'), ('C1', 'C2'), ('C0', 'C2')):
            src = series[origin]
            r = transfer(series[destination], src, f'{origin}->{destination}')
            r.update(account=origin,native_G=src['G'],native_J=src['J'],
                     native_W_v=src['W_v'],native_W_f=src['W_f'],
                     native_residual_v=src['residual_capacity_v'],native_residual_f=src['residual_capacity_f'],
                     native_util_v=src['utilization_v'],native_util_f=src['utilization_f'])
            m.append(r)
    df=pd.DataFrame(m);df.to_csv(RESULTS/'mandatory_accounts_transfers.csv',index=False)
    m=[]
    for _,g in t[t.experiment=='mandatory_elasticity'].groupby(list(KEYS)+['alpha']):
        base=g[g.elasticity==0].iloc[0].to_dict()
        for _,z in g.iterrows():
            row=transfer(z.to_dict(),base,'frozen-to-responsive')
            row.update(E=z.E,access=z.access,staff=z.k_v,value=z.J,delay=z.mandatory_minutes,
                       gap=z.gap,p_v=z.p_v,p_f=z.p_f)
            m.append(row)
    df2=pd.DataFrame(m);df2.to_csv(RESULTS/'elasticity_transfers.csv',index=False)
    return df,df2


def rational(x):
    x=float(x)
    p=F(4160,147)
    if abs(x-float(p))<1e-12:
        return p
    if abs(x-float(F(1,3)))<1e-12:
        return F(1,3)
    return F(str(x))


def exact_case(c):
    return exact.Case(a=rational(c.a),b=rational(c.b if c.path!='N' else 0),d=rational(c.d),
                     theta=rational(c.theta),demand=rational(c.demand),required=rational(c.required),
                     V=rational(c.V),premium=rational(c.s)-rational(c.T),
                     cost=(rational(c.cv),rational(c.cf)),rates=(rational(c.nu_v),rational(c.nu_f)),
                     alpha=rational(c.alpha),r=rational(c.penalty),
                     cap=(rational(c.box[1]),rational(c.box[3])),K=int(c.K))


def cap_exact(c):
    q=exact_case(c); S=c.path=='S'
    pv,pf=q.cap
    bv,bf,be=exact.bills(q,q.cap,S)
    both=[exact.hits(q,i,S) for i in (0,1)]
    basic='in-person' if bf>bv and both[1] else 'virtual' if bv>bf and both[0] else ''
    aided=''
    if S and bf+q.r>0 and both[1]:
        L=(bf+q.r)*q.demand/(max(bv,bf)+q.r)
        loads=(q.required*q.d+q.b*L,q.required+q.a*L)
        time_floor=(1+q.a*q.b)*q.cost[0]/q.rates[0]+q.a*q.cost[1]/q.rates[1]
        delays=[]
        for k in range(1,q.K):
            try:
                delay=(1+q.a*q.b)*q.cost[0]*exact.wait(loads[0],k,q.rates[0])
                if q.a:
                    delay+=q.a*q.cost[1]*exact.wait(loads[1],q.K-k,q.rates[1])
                if loads[1]>=(q.K-k)*q.rates[1]:
                    continue
                delays.append(delay)
            except ValueError:
                continue
        floor=max(time_floor,min(delays)) if delays else time_floor
        if min(bv,(q.values[0]-floor)/q.alpha)<bf:
            aided='in-person'
    return q,bv,bf,be,both,basic,aided


def caps(t):
    z=t[t.experiment.isin(['paths','policy_switch','exact_small_role','exact_small_gain'])]
    rows=[];family=[]
    for _,v in z.iterrows():
        c=sc_of(v)
        q,bv,bf,be,xx,basic,strong=cap_exact(c)
        row={x:v[x] for x in ('scenario_id','experiment','path',*KEYS,'alpha')}
        row.update(pi_v=float(bv),pi_f=float(bf),V_cap_hits='|'.join(str(p['k'][0]) for p in xx[0]),
                   F_cap_hits='|'.join(str(p['k'][0]) for p in xx[1]),
                   T3_dominance=basic,T3_strong_screen=strong)
        i=1 if strong else 0 if basic=='virtual' else 1 if basic else -1
        if c.regime=='F0' and i<0 and xx[1]:
            det=1 if c.path=='S' else 1-q.a*q.b
            paid=1 if c.path=='S' else 1-q.theta*q.a*q.b
            if q.n*q.V-q.alpha*paid*q.cap[0]-det*q.cost[0]/q.rates[0]<0:
                row['F0_virtual_excluded']=True
                i=1
        if i>=0:
            J=q.demand*(bv,bf)[i]+q.required*be
            assert v['lower_bound']-1e-6<=float(J)<=v['upper_bound']+1e-6
            row.update(exact_cap_value=float(J),opt_k='|'.join(str(p['k'][0]) for p in xx[i]))
            for p in xx[i]:
                family.append(dict(**row,k_v=p['k'][0],k_f=p['k'][1],
                                   p_v=float(p['p'][0]),p_f=float(p['p'][1]),
                                   x_v=float(p['x'][0]),x_f=float(p['x'][1]),
                                   W_v=float(p['W'][0]),W_f=float(p['W'][1]),
                                   D_E=480*float(p['W'][1]+q.d*p['W'][0]),
                                   U_v=float(p['U'][0]),U_f=float(p['U'][1])))
        elif bv==bf and (xx[0] or xx[1]):
            row['exact_cap_value']=float(q.demand*bv+q.required*be)
            row['pure_witnesses_only']=True
            for p in xx[0]+xx[1]:
                family.append(dict(**row,k_v=p['k'][0],k_f=p['k'][1],
                                   p_v=float(p['p'][0]),p_f=float(p['p'][1]),
                                   x_v=float(p['x'][0]),x_f=float(p['x'][1]),
                                   W_v=float(p['W'][0]),W_f=float(p['W'][1]),
                                   D_E=480*float(p['W'][1]+q.d*p['W'][0]),
                                   U_v=float(p['U'][0]),U_f=float(p['U'][1])))
        rows.append(row)
    d=pd.DataFrame(rows); d.to_csv(RESULTS/'cap_implementation.csv',index=False)
    family=pd.DataFrame(family)
    family.to_csv(RESULTS/'exact_cap_policy_family.csv',index=False)
    return d,family


def tie_transfers(t,caps,family):
    # If both coordinates are billed, every optimal tariff is at the cap.
    # Otherwise compute a lower/upper enclosure for the best and worst losses
    # over the true optimum set using a universal revenue bound and sample witnesses.
    p=t[t.experiment=='paths']
    rows=[]
    exact_o=family[(family.experiment=='paths') & (family.path=='O')]
    for key,g in p.groupby(list(KEYS),sort=False):
        s=g[g.path=='S'].iloc[0].to_dict();o=g[g.path=='O'].iloc[0].to_dict()
        m=exact_o[exact_o.scenario_id==o['scenario_id']]
        if not m.empty:
            for name,val in zip(KEYS,key):
                m=m[m[name]==val]
        fixed=len(m)>0
        witnesses=[]
        for _,v in m.iterrows():
            sc=sc_of(o)
            _,H,_,he,_=sc.visits
            mass=H@np.array([v.x_v,v.x_f])+sc.required*he
            fixed=fixed and all(mi>0 or lo==high for mi,lo,high in zip(mass,[sc.box[0],sc.box[2]],[sc.box[1],sc.box[3]]))
            w=evaluate(sc_of(s),int(v.k_v),v.p_v,v.p_f)
            if w is not None:
                witnesses.append(loss(s,w))
        # If there is no exact proof of eligibility, the reported policy is an
        # epsilon-near-optimal witness only, so no best/worst tightening is asserted.
        if witnesses:
            complete=fixed and not m.get('pure_witnesses_only', pd.Series(False, index=m.index)).eq(True).any()
            row=dict(best_lo=min(z['loss_lower'] for z in witnesses) if complete else 0,
                     best_hi=min(z['loss_upper'] for z in witnesses),
                     worst_lo=max(z['loss_lower'] for z in witnesses),
                     worst_hi=max(z['loss_upper'] for z in witnesses) if complete else s[UL]+s['penalty']*s['demand'],
                     class_status='all_exact_tariffs' if complete else 'exact_witnesses_plus_bounds',samples=len(witnesses))
        else:
            row=dict(best_lo=0,best_hi=s[UL]+s['penalty']*s['demand'],worst_lo=0,
                     worst_hi=s[UL]+s['penalty']*s['demand'],class_status='universal_bounds',samples=0)
        row.update(dict(zip(KEYS,key)),O_id=o['scenario_id'],S_id=s['scenario_id'])
        rows.append(row)
    tbl=pd.DataFrame(rows);tbl.to_csv(RESULTS/'tie_transfer_bounds.csv',index=False)
    return tbl


def impacts(t):
    a=[]
    def subtract(b,c,kind):
        dv=480*(c['W_v']-b['W_v']);df=480*(c['W_f']-b['W_f'])
        cutoff=-df/dv if dv else np.nan
        low=bool(df<0 and df+dv>0)
        high=bool(df>0 and df+dv<0)
        a.append(dict(kind=kind,demand=c['demand'],s=c['s'],theta=c['theta'],regime=c['regime'],
                      b_before=b['b'],b_after=c['b'],k_before=b['k_v'],k_after=c['k_v'],
                      id_before=b['scenario_id'],id_after=c['scenario_id'],delta_v=dv,delta_f=df,
                      d_critical=cutoff,delta_d0=df,delta_d1=(df+dv),opposite=low or high,
                      both_unique=bool(c['unique_optimal_k'] and b['unique_optimal_k'])))
    v=t[t.experiment=='followup_scan']
    for _,g in v.groupby(['demand','s','theta','regime']):
        rows=g.sort_values('b').to_dict('records')
        for b,c in zip(rows,rows[1:]):
            if b['k_v']!=c['k_v']:
                subtract(b,c,'reference')
    q=t[t.experiment=='policy_switch']
    l=q[q.b==.245].iloc[0].to_dict();r=q[q.b==.255].iloc[0].to_dict()
    l['unique_optimal_k']=True; r['unique_optimal_k']=True  # exact end points in scale_witness
    subtract(l,r,'exact-cross-point')
    selected=exact.scale_witness()
    v=selected['threshold_virtual'];f=selected['threshold_in_person']
    x=q[q.b==.25].iloc[0].to_dict()
    p=dict(x,W_v=float(v['W'][0]),W_f=float(v['W'][1]),k_v=13,unique_optimal_k=False)
    y=dict(x,W_v=float(f['W'][0]),W_f=float(f['W'][1]),k_v=3,unique_optimal_k=False)
    subtract(p,y,'exact-common-threshold')
    df=pd.DataFrame(a);df.to_csv(RESULTS/'patient_impacts.csv',index=False)
    return df



def threshold_family(t):
    v=t[(t.experiment=='policy_switch')&(t.b==.25)].iloc[0]
    sc=sc_of(v); c=exact_case(sc)
    cap=exact.capval(c)
    rows=[]
    for k in range(1,c.K):
        candidate=evaluate(sc,k,*map(float,c.cap))
        label='undetermined'
        if candidate['point_upper']<float(cap)-1e-8:
            label='excluded'
        else:
            pure=[z for j in range(2) for z in exact.hits(c,j) if z['k'][0]==k]
            if pure:
                label='exact_pure'
            elif candidate['entry']=='both':
                x=F(str(candidate['x_v']))
                half=F('0.000001')
                try:
                    a=exact.policy(c,k,(x-half,c.demand-x+half))
                    b=exact.policy(c,k,(x+half,c.demand-x-half))
                except ValueError:
                    pass
                else:
                    if a['U'][0]>a['U'][1] and b['U'][0]<b['U'][1]:
                        max_v=b['W'][0];max_f=a['W'][1]
                        qv=1+c.a*c.b; bv,bf,_=exact.bills(c,c.cap)
                        min_u=[c.values[0]-c.alpha*bv-qv*c.cost[0]*max_v-c.a*c.cost[1]*max_f,
                               c.values[1]-c.alpha*bf-c.b*c.cost[0]*max_v-c.cost[1]*max_f]
                        if min(min_u)>0:
                            label='exact_full_mixed'
            if label.startswith('exact'):
                assert candidate['point_upper']>=float(cap)-1e-8
        rows.append(dict(k_v=k,status=label,J_cap=float(cap),cap_point_lower=candidate['lower_bound'],
                         cap_point_upper=candidate['point_upper'],**{j:candidate[j] for j in
                         ['x_v','x_f','W_v','W_f','U_v','U_f','mandatory_minutes']}))
    pd.DataFrame(rows).to_csv(RESULTS/'exact_design_tie_set.csv',index=False)
    return rows

def main():
    t=pd.read_csv(RESULTS/'experiments.csv').drop_duplicates(
        ['experiment','scenario_id','path','a','b','d','theta','gamma','elasticity','alpha',
         'varied','account','witness'],keep='first')
    assert (t.stable==True).all()
    st=pd.read_csv(RESULTS/'staff_bounds.csv')
    s,p=path_analyses(t)
    k=price_only(t)
    r=recurrent(t)
    c,e=mandatory(t,st)
    cap,fam=caps(t)
    ties=tie_transfers(t,cap,fam)
    imp=impacts(t)
    threshold_family(t)
    scan=t[t.experiment=='followup_scan']
    ab=t[t.experiment=='independent_ab']
    exact_script=RESULTS/'exact_checks.json'
    payload={'role_change':exact.role_change(),'latent_gain':exact.latent_gain(),'partial':exact.partial(),
             'switch':exact.switch(),'scale_screen':exact.scale_screen(),'scale_witness':exact.scale_witness()}
    exact_script.write_text(json.dumps(payload,default=exact.enc,indent=2)+'\n')
    counts=t.groupby('experiment').agg(N=('J','size'),tol=('gap',lambda s:sum(s<=.1)),
            gap_max=('gap','max'),gap_med=('gap','median'),seconds_med=('elapsed','median'),
            seconds_max=('elapsed','max'),relative_max=('relative_gap','max'),residual_max=('wardrop_residual','max'))
    counts.to_csv(RESULTS/'solver_summary.csv')
    # Same-price F0 cases use the same strategic and compulsory paths at b=0 by algebra.
    assert len(ab)==324 and sum(w['both'] for w in payload['scale_screen'])==4
    selected=ab[(ab.clinician_time_difference_minutes<0)&(ab.entry=='in-person')]
    checks=dict(total_rows=len(t),unique_searches=int(t.scenario_id.nunique()),
                tolerance_rows=int((t['gap']<=.1).sum()),remaining_rows=int((t['gap']>.1).sum()),
                worst_gap=float(t['gap'].max()),median_seconds=float(t.elapsed.median()),
                max_seconds=float(t.elapsed.max()),max_residual=float(t.wardrop_residual.max()),
                path_transfers=len(s),cycle_transfers=len(r),accounts=len(c),elasticity=len(e),
                grid_ab=len(ab),grid_ab_efficiency_in_person=len(selected),
                scaled_threshold_passes=4,
                design_strict_repricing_loss=k[(k.alpha==.3)&(k.b==.255)][
                  ['b','k_v','k_v_fixed','J','J_fixed','loss_lower','loss_upper']].iloc[0].to_dict(),
                v2_reference_switches=int((imp.kind=='reference').sum()),
                v2_reference_opposite=int(imp[imp.kind=='reference'].opposite.sum()),
                v2_reference_unique=int(imp[imp.kind=='reference'].both_unique.sum()),
                cap_S_passes=int(cap[(cap.experiment=='paths')&(cap.path=='S')].exact_cap_value.count()),
                raw_roles=t[(t.experiment=='paths')&(t.path=='S')].entry.value_counts().to_dict())
    (RESULTS/'analysis_summary.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))
    print(counts.to_string())


if __name__=='__main__':
    main()
