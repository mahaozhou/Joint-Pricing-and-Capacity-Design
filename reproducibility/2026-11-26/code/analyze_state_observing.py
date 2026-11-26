"""Summarize 100 paired replications per allocation; independently verify analytical baseline.
Run with python code/analyze_state_observing.py. Use --data-dir for a separate run
directory and --figure-dir to request the optional PDF and PNG plots.
"""
from argparse import ArgumentParser
from pathlib import Path
import math,json
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "results" / "state_observing"
parser = ArgumentParser(description=__doc__)
parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
parser.add_argument("--figure-dir", type=Path, help="write optional plots to this directory")
args = parser.parse_args()
D = args.data_dir
r=pd.read_csv(D/'replications.csv'); a=pd.read_csv(D/'source_exact_design_tie_set.csv').query('3 <= k_v <= 13').set_index('k_v')
assert len(r)==2200 and r.groupby(['kv','mode']).size().eq(100).all()
assert not r.duplicated(['kv','rep','mode']).any()
assert (r.potential==r.virtual+r.inperson+r.balk).all() and (r.mandatory==r.completed_mandatory).all()
assert np.allclose(r.mandatory_time,r.mandatory_initial_time+r.mandatory_followup_time,rtol=1e-9,atol=1e-9)
assert (r.groupby('rep').seed.nunique()==1).all()
assert (r.groupby('rep').potential.nunique()==1).all() and (r.groupby('rep').mandatory.nunique()==1).all()
# Analytical M/M/k check, independently of the archived W values.
L,E,d1,d2,dE,theta,alpha,V,s,T,cv,cf,pv,pf,nu,pen=572,30,.35,.25,.25,.5,.3,50,15,20,112,168,4160/147,40,54,10
Bf=pf+theta*d2*pv; Bv=pv+d1*Bf; BE=pf+theta*dE*pv
assert abs(Bv-Bf)<1e-12

def W(load,k):
 z=load/nu; assert z<k
 term=z**k/math.factorial(k)/(1-z/k)
 C=term/(sum(z**i/math.factorial(i) for i in range(k))+term)
 return 1/nu+C/(k*nu-load)

pol=[]
for k,x in a.iterrows():
 xv,xf=x.x_v,x.x_f
 lv=(1+d1*d2)*xv+d2*xf+dE*E; lf=d1*xv+xf+E
 wv,wf=W(lv,k),W(lf,18-k)
 uv=V+d1*(s-T)-alpha*Bv-(1+d1*d2)*cv*wv-d1*cf*wf
 uf=V+s-T-alpha*Bf-d2*cv*wv-cf*wf
 j=xv*Bv+xf*Bf+E*BE-pen*(L-xv-xf); de=480*(wf+dE*wv)
 assert np.allclose([wv,wf,uv,uf,j,de],[x.W_v,x.W_f,x.U_v,x.U_f,x.J_cap,x.mandatory_minutes],atol=1e-7,rtol=1e-9)
 assert abs(xv+xf-L)<1e-7 and max(uv,uf)>0
 assert (xv==0 or uv>=uf-1e-7) and (xf==0 or uf>=uv-1e-7)
 pol.append(dict(kv=k,kf=18-k,pv=pv,pf=pf,xv=xv,xf=xf,load_v=lv,load_f=lf,util_v=lv/(k*nu),util_f=lf/((18-k)*nu),Wv=wv,Wf=wf,Uv=uv,Uf=uf,J=j,DE_min=de,virtual_share_pp=100*xv/L))
pd.DataFrame(pol).to_csv(D/'analytical_policies.csv',index=False)

def ci(x):
 x=np.asarray(x,dtype=float); m=x.mean(); h=t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x));return m,m-h,m+h
metrics=['J','J_realized','access_pp','virtual_share_pp','DE_min']
rows=[]; paired=[]; checks=[]; pairraw=[]
for k,g in r.groupby('kv'):
 b=g[g['mode']==0].set_index('rep').sort_index(); o=g[g['mode']==1].set_index('rep').sort_index(); assert (b.seed==o.seed).all()
 d=o[metrics]-b[metrics]; d.insert(0,'kv',k); pairraw.append(d.reset_index())
 for label,x in [('baseline',b),('state_observing',o),('paired_difference',d)]:
  for m in metrics:
   mu,lo,hi=ci(x[m]); rows.append(dict(kv=k,condition=label,metric=m,mean=mu,lower95=lo,upper95=hi,n=100))
 z=dict(kv=k,baseline_virtual_share=b.virtual_share_pp.mean(),observed_virtual_share=o.virtual_share_pp.mean(),baseline_DE=b.DE_min.mean(),observed_DE=o.DE_min.mean())
 for m in metrics:
  mu,lo,hi=ci(d[m]); z.update({f'delta_{m}':mu,f'delta_{m}_lo':lo,f'delta_{m}_hi':hi})
 paired.append(z)
 for m,truth in [('DE_min',a.loc[k,'mandatory_minutes']),('virtual_share_pp',100*a.loc[k,'x_v']/L),('J',a.loc[k,'J_cap']),('J_realized',a.loc[k,'J_cap'])]:
  mu,lo,hi=ci(b[m]);checks.append(dict(kv=k,metric=m,analytical=truth,mean=mu,lo=lo,hi=hi,covered=lo-1e-10<=truth<=hi+1e-10))
sumdf=pd.DataFrame(rows); sumdf.to_csv(D/'summary_long.csv',index=False)
s=pd.DataFrame(paired); s.to_csv(D/'paired_summary.csv',index=False);pd.concat(pairraw).to_csv(D/'paired_replications.csv',index=False)
checks=pd.DataFrame(checks);checks.to_csv(D/'baseline_checks.csv',index=False)
audit={'replications_per_policy_per_rule':100,'policy_count':11,'simulation_runs':len(r),'total_events':int(r.events.sum()),'strategic_arrivals_per_rule':{str(k):int(v) for k,v in r.groupby('mode').potential.sum().items()},'balks_per_rule':{str(k):int(v) for k,v in r.groupby('mode').balk.sum().items()},'baseline_CI_coverage':{m:int(g.covered.sum()) for m,g in checks.groupby('metric')},'max_abs_paired_expected_J':float(s.delta_J.abs().max()),'realized_J_difference_CI_contains_zero':int(((s.delta_J_realized_lo<=0)&(s.delta_J_realized_hi>=0)).sum()),'all_completed':True,'runtime_log':(D/'run.log').read_text()}
(D/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2)); print(s[['kv','baseline_virtual_share','observed_virtual_share','baseline_DE','observed_DE','delta_DE_min','delta_DE_min_lo','delta_DE_min_hi']].round(3).to_string(index=False))
with (D / "table_G1_rows.tex").open("w", encoding="utf-8", newline="\n") as table:
    for row in paired:
        values = [
            str(row["kv"]),
            f'{row["baseline_virtual_share"]:.2f}',
            f'{row["observed_virtual_share"]:.2f}',
            f'{row["baseline_DE"]:.2f}',
            f'{row["observed_DE"]:.2f}',
        ]
        change = row["delta_DE_min"]
        lo = row["delta_DE_min_lo"]
        hi = row["delta_DE_min_hi"]
        values.append(fr'${change:.2f}\ [{lo:.2f}, {hi:.2f}]$')
        table.write(" & ".join(values) + r" \\" + "\n")

if args.figure_dir is not None:
    F = args.figure_dir
    F.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(1,2,figsize=(7.0,3.0),layout='constrained')
    for a0,base,obs,title,ylabel in [(ax[0],'baseline_virtual_share','observed_virtual_share','(a) Initial-channel choice','Virtual-first share (%)'),(ax[1],'baseline_DE','observed_DE','(b) Mandatory patients','Episode sojourn time (min)')]:
     a0.plot(s.kv,s[base],marker='o',mfc='white',color='#6c747a',label='Baseline');a0.plot(s.kv,s[obs],marker='s',color='#2b526c',label='State-observing');a0.set(xlabel=r'Virtual clinicians $k_v$',ylabel=ylabel,title=title,xticks=list(range(3,14,2)));a0.grid(axis='y',alpha=.18)
    ax[0].legend(frameon=False,loc='upper left',fontsize=9);ax[1].legend(frameon=False,loc='upper right',fontsize=9)
    fig.savefig(F/'figG1_state_observing.pdf',bbox_inches='tight');fig.savefig(F/'figG1_state_observing.png',dpi=200,bbox_inches='tight');plt.close(fig)

# Utilization and queue diagnostics, including the extended run for k_v=3.
util=r.groupby(['kv','mode']).agg(util_v_mean=('util_v','mean'),util_f_mean=('util_f','mean'),max_N_v=('max_N_v','max'),max_N_f=('max_N_f','max'),entries_v_mean=('entries_v','mean'),entries_f_mean=('entries_f','mean')).reset_index()
util.to_csv(D/'utilization_diagnostics.csv',index=False)
e=pd.read_csv(D/'extended_k3.csv'); assert len(e)==200 and e.groupby('mode').size().eq(100).all() and (e.mandatory==e.completed_mandatory).all()
assert (e.groupby('rep').potential.nunique()==1).all() and (e.groupby('rep').mandatory.nunique()==1).all()
base=e[e['mode']==0].set_index('rep'); obs=e[e['mode']==1].set_index('rep'); extended=[]
for name,x in [('baseline',base),('state_observing',obs),('paired_difference',obs[metrics]-base[metrics])]:
 for m in metrics:
  mu,lo,hi=ci(x[m]);extended.append(dict(condition=name,metric=m,mean=mu,lower95=lo,upper95=hi,n=100))
pd.DataFrame(extended).to_csv(D/'extended_k3_summary.csv',index=False)
audit.update({'extended_runs':len(e),'extended_balks':int(e.balk.sum()),'extended_events':int(e.events.sum()),'extended_runtime_log':(D/'extended.log').read_text(),'extended_k3_delta_DE':ci(obs.DE_min-base.DE_min),'extended_k3_state_virtual_utilization':float(obs.util_v.mean())})
(D/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
