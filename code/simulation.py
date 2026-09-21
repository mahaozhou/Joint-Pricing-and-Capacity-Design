"""Patient-level FCFS DES: staged pathways, k servers, episode cohort follow-through.

All final evaluations use 10,000 independent replications. The measurement window
is 20 clinic-day units following 10 warm-up units. Arrivals continue until every
enrolled episode has completed, avoiding administrative truncation of referrals.
"""
from pathlib import Path
from dataclasses import asdict
import json, argparse, time
import numpy as np
from numba import njit, prange, set_num_threads
from scipy.stats import t as student_t
from model import Model, evaluate
from experiments import solve_model

from paths import SIMULATION as OUT
REPLICATIONS=10000

@njit(cache=True)
def one_rep(seed,L,E,d1,d2,gamma,de,caps,ks,prices,theta,copay,costs,v,q,b,
            fractions,forecast,penalty,warm,duration,mode):
    np.random.seed(seed)
    end=warm+duration
    # Queue entries need only retain patient, stage, and node-arrival timestamp.
    size=int((L+E)*(end+50)*1.5+10000)
    queue_pid=np.zeros((2,size),np.int64)
    queue_stage=np.zeros((2,size),np.int64)
    queue_time=np.zeros((2,size))
    head=np.zeros(2,np.int64);tail=np.zeros(2,np.int64)
    nserver=int(ks.sum())
    node=np.zeros(nserver,np.int64);node[int(ks[0]):]=1
    finish=np.ones(nserver)*np.inf
    spid=np.zeros(nserver,np.int64);sstage=np.zeros(nserver,np.int64)
    sarr=np.zeros(nserver)
    origin=np.zeros(size,np.int64);birth=np.zeros(size)
    enrolled=np.zeros(size,np.bool_);burden=np.zeros(size)
    first_time=np.zeros(size)
    sums=np.zeros(3);counts=np.zeros(3);costsum=np.zeros(3)
    encounter_sum=np.zeros(4);encounter_count=np.zeros(4)
    integral=np.zeros(2)
    nextS=np.random.exponential(1/L) if L>0 else np.inf
    nextE=np.random.exponential(1/E) if E>0 else np.inf
    now=0.; pid=0; remaining=0
    accepted=0.;total=0.;balk=0.;gross=0.;events=0
    while now<end or remaining>0:
        si=np.argmin(finish)
        nxt=min(nextS,nextE,finish[si])
        if nxt> end+200 or pid>=size-3 or max(tail)>=size-3:
            return np.ones(22)*np.nan
        dt=max(0.,min(nxt,end)-max(now,warm))
        current=np.zeros(2)
        for h in range(2):
            busy=0
            for s in range(nserver):
                if node[s]==h and np.isfinite(finish[s]):busy+=1
            n=tail[h]-head[h]+busy
            current[h]=ks[h]/caps[h]+max(0,n-ks[h]+1)/caps[h]
            integral[h]+=dt*current[h]
        now=nxt;events+=1
        addnode=-1;addpid=-1;addstage=0
        if now==nextS or now==nextE:
            isE=now==nextE
            if isE:
                nextE=now+np.random.exponential(1/E)
                selected=1;typ=2
            else:
                nextS=now+np.random.exponential(1/L)
                if mode==0:
                    rr=np.random.random()
                    selected=0 if rr<fractions[0] else (1 if rr<sum(fractions) else -1)
                else:
                    u=v-copay*b.T@prices-q.T@(costs*forecast)
                    # Initial state is observed; future encounters use advertised means.
                    u[0]-=costs[0]*(current[0]-forecast[0])
                    u[1]-=costs[1]*(current[1]-forecast[1])
                    selected=-1 if max(u)<=0 else (0 if u[0]>u[1] else 1)
                typ=selected
                if warm<=now<end:
                    total+=1
                    if selected>=0:accepted+=1
                    else:balk+=1
            if selected>=0:
                pid+=1;origin[pid]=typ;birth[pid]=now
                if warm<=now<end:
                    enrolled[pid]=True;remaining+=1
                addnode=selected;addpid=pid
        else:
            h=node[si];p=spid[si];stage=sstage[si]
            sojourn=now-sarr[si]
            burden[p]+=costs[h]*sojourn
            if warm<=sarr[si]<end:
                kind=2*h+(1 if stage>0 else 0)
                encounter_sum[kind]+=sojourn;encounter_count[kind]+=1
            if warm<=now<end:
                gross+=prices[h]*(1. if stage==0 else theta)
            finish[si]=np.inf
            if stage==0:
                probability=d1 if h==0 else (de if origin[p]==2 else d2)
                if np.random.random()<probability:
                    addnode=1-h;addpid=p;addstage=1
            elif stage==1 and origin[p]==0 and h==1:
                if np.random.random()<gamma:
                    addnode=0;addpid=p;addstage=2
            if addnode<0 and enrolled[p]:
                typ=origin[p];sums[typ]+=now-birth[p];counts[typ]+=1
                costsum[typ]+=burden[p];remaining-=1
            # FCFS start next waiting encounter on the newly freed server.
            if head[h]<tail[h]:
                idx=head[h];head[h]+=1
                spid[si]=queue_pid[h,idx];sstage[si]=queue_stage[h,idx]
                sarr[si]=queue_time[h,idx]
                finish[si]=now+np.random.exponential(ks[h]/caps[h])
        if addnode>=0:
            h=addnode;free=-1
            for s in range(nserver):
                if node[s]==h and not np.isfinite(finish[s]):
                    free=s;break
            if free>=0:
                spid[free]=addpid;sstage[free]=addstage;sarr[free]=now
                finish[free]=now+np.random.exponential(ks[h]/caps[h])
            else:
                idx=tail[h];tail[h]+=1
                queue_pid[h,idx]=addpid;queue_stage[h,idx]=addstage;queue_time[h,idx]=now
    result=np.empty(22)
    result[0]=gross/duration;result[1]=(gross-penalty*balk)/duration
    result[2]=accepted/max(total,1.)
    result[3:6]=sums/np.maximum(counts,1)
    result[6:9]=counts
    result[9:12]=costsum/np.maximum(counts,1)
    result[12:16]=encounter_sum/np.maximum(encounter_count,1)
    result[16:20]=encounter_count
    result[20:22]=integral/duration
    return result

@njit(cache=True,parallel=True)
def many(reps,seed,L,E,d1,d2,gamma,de,caps,ks,prices,theta,copay,costs,v,q,b,
         fractions,forecast,penalty,warm,duration,mode):
    values=np.zeros((reps,22))
    for r in prange(reps):
        values[r]=one_rep(seed+r,L,E,d1,d2,gamma,de,caps,ks,prices,theta,copay,costs,v,q,b,
         fractions,forecast,penalty,warm,duration,mode)
    return values

def summarize_values(vals):
    reps=len(vals)
    names=['gross','objective','access','episode_on','episode_off','episode_E','count_on','count_off','count_E',
           'cost_on','cost_off','cost_E','encounter_on_first','encounter_on_followup','encounter_off_first',
           'encounter_off_followup','n_on_first','n_on_followup','n_off_first','n_off_followup','virtual_on','virtual_off']
    summary={}
    for i,n in enumerate(names):
        sample=vals[:,i]
        # An unobserved class has no delay estimate.
        mask=np.ones(reps,dtype=bool)
        if i in [3,4,5,9,10,11]: mask=vals[:,6+(i-3)%3]>0
        if 12<=i<=15: mask=vals[:,i+4]>0
        if not np.any(mask): summary[n]=None;continue
        weights=None
        if i in [3,4,5,9,10,11]: weights=vals[:,6+(i-3)%3]
        if 12<=i<=15: weights=vals[:,i+4]
        if weights is not None:
            # Ratio of total time to total encounters/episodes; independent
            # replication clusters provide the delta-method standard error.
            mean=float(np.sum(sample*weights)/np.sum(weights))
            influence=(sample-mean)*weights/np.mean(weights)
            h=float(student_t.ppf(.975,reps-1)*np.std(influence,ddof=1)/np.sqrt(reps))
            summary[n]=dict(mean=mean,halfwidth=h,n=reps,total_observations=int(np.sum(weights)))
        else:
            h=float(student_t.ppf(.975,reps-1)*np.std(sample,ddof=1)/np.sqrt(reps))
            summary[n]=dict(mean=float(np.mean(sample)),halfwidth=h,n=reps)
    return summary

def simulate(m,r,mode=0,reps=REPLICATIONS,seed=910000,warm=10.,duration=20.):
    if m.recurrent:raise ValueError('DES implements the staged pathway; recurrent results are analytical only')
    q,e,b,be,v=m.matrices()
    caps=np.array([r['mu_on'],r['mu_off']]);prices=np.array([r['p_on'],r['p_off']])
    ks=np.array([int(round(r['decision'])),m.servers-int(round(r['decision']))]) if m.servers else np.ones(2,dtype=np.int64)
    vals=many(reps,seed,float(m.demand),r['required_actual'],m.d1,m.d2,m.d2 if m.gamma<0 else m.gamma,
        m.d2 if m.dE<0 else m.dE,caps,ks,prices,m.followup_fee,m.copay,np.array([m.cost_on,m.cost_off]),v,q,b,
        np.array([r['flow_on'],r['flow_off']])/m.demand,np.array([r['wait_on'],r['wait_off']]),m.penalty,warm,duration,mode)
    if np.any(~np.isfinite(vals)):raise RuntimeError('DES overflow or excessive terminal continuation')
    summary=summarize_values(vals)
    return vals,dict(model=asdict(m),policy=r,mode='wardrop' if mode==0 else 'state_observing',
        replications=reps,seed=seed,warmup=warm,measurement=duration,statistics=summary)

def cache_matches(record,m,r,mode=0,reps=REPLICATIONS,seed=910000,warm=10.,duration=20.):
    if record.get('model')!=asdict(m):return False
    expected_mode='wardrop' if mode==0 else 'state_observing'
    if any(record.get(key)!=value for key,value in (
        ('mode',expected_mode),('replications',reps),('seed',seed),
        ('warmup',warm),('measurement',duration))):return False
    old=record.get('policy',{})
    keys=('decision','p_on','p_off','objective','flow_on','flow_off')
    return all(key in old and abs(float(old[key])-float(r[key]))<=1e-6 for key in keys)

def run():
    set_num_threads(4)
    records=[]
    cases=[('moderate',Model()),('high',Model(demand=878)),
           ('mixed',Model(demand=572,premium=15,d2=.4)),
           ('mixed_high',Model(demand=878,premium=15,d2=.4))]
    for name,m in cases:
        path=OUT/f'des_{name}.json'
        r=solve_model(m,'dual')
        if path.exists():
            cached=json.loads(path.read_text())
            if cache_matches(cached,m,r):records.append(cached);continue
        start=time.time();vals,summary=simulate(m,r)
        summary['name']=name;summary['seconds']=time.time()-start
        np.savez_compressed(OUT/f'des_{name}_replications.npz',values=vals)
        path.write_text(json.dumps(summary,indent=2));records.append(summary)
        print(name,r['region'],'10000 replications',round(summary['seconds'],1),'seconds',flush=True)
    (OUT/'simulation.json').write_text(json.dumps(records,indent=2))

if __name__=='__main__':run()
