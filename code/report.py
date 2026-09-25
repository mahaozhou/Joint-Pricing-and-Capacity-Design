"""Build every numerical exhibit from saved outputs; no hand-entered results."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
import pandas as pd
from scipy.stats import t as student_t
from shorten_captions import short_caption, clean_asset

from paths import MAIN, STAFFING, SIMULATION, STATE, TABLES, FIGURES as FIG
OUT=MAIN
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
    'axes.spines.right':False,'pdf.fonttype':42,'axes.labelsize':9,'legend.fontsize':8})
def read(name):return json.loads((OUT/name).read_text())
def write(name,text):
    if name in {'referral_text.tex','required_text.tex','staffing_text.tex',
                'pathway_text.tex','simulation_text.tex','reserve_text.tex'}:
        text='% Narrative is maintained in the manuscript and supplement.\n'
    (TABLES/name).write_text(clean_asset(text),encoding='utf-8')
def save(fig,name):
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(FIG/f'{name}.png',dpi=170,bbox_inches='tight');plt.close(fig)
def number(x,n=1):return f'{x:,.{n}f}'
DELAY_LABELS = {
    'episode_on': 'Virtual-first episode',
    'episode_off': 'In-person-first episode',
    'episode_E': 'Exogenous episode',
    'encounter_on_first': 'Initial virtual encounter',
    'encounter_on_followup': 'Virtual follow-up',
    'encounter_off_first': 'Initial in-person encounter',
    'encounter_off_followup': 'Referred in-person encounter',
}
def table(caption,label,cols,headers,rows,size='small'):
    caption=short_caption(caption)
    if size=='scriptsize':size='footnotesize'
    return '\\begin{table}[ht]\n\\centering\\'+size+'\n\\caption{'+caption+'}\\label{'+label+'}\n\\begin{tabular}{@{}'+cols+'@{}}\\toprule\n'+ ' & '.join(headers)+r'\\\midrule'+'\n'+'\n'.join(' & '.join(row)+r'\\' for row in rows)+'\n\\bottomrule\\end{tabular}\n\\end{table}\n'

def build():
    rows=read('experiments.json');transfers=read('transfers.json');staff=json.loads((STAFFING/'staffing_transfers.json').read_text())
    aware=read('aware_transfers.json') if (OUT/'aware_transfers.json').exists() else []
    sim=json.loads((SIMULATION/'validation_all.json').read_text()) if (SIMULATION/'validation_all.json').exists() else json.loads((SIMULATION/'simulation.json').read_text())
    regime={'fixed':'F0','online':'F1','dual':'F2'}
    def case_name(name):
        return {
            'prescribed_VF':'constructed VF',
            'prescribed_BVF':'constructed BVF',
            'mixed':'virtual-only',
            'mixed_high':'partial virtual',
            'state_max_offline':'maximum in-person price',
        }.get(name,name.replace('state_','').replace('_',' '))
    def select(group,**kwargs):
        return [r for r in rows if r['group']==group and all(r['model'][k]==v for k,v in kwargs.items())]
    macro={'ExperimentCount':len(rows),'TransferCount':len(transfers),'UnstableTransfers':sum(t['loss'] is None for t in transfers)}
    write('summary_macros.tex','\n'.join('\\newcommand{\\'+k+'}{'+str(v)+'}' for k,v in macro.items()))
    # Export all reported policies as a flat CSV in addition to full JSON configurations.
    pd.DataFrame([{**{k:r[k] for k in ['group','label','regime']},**r['model'],**r['result']} for r in rows]).to_csv(OUT/'all_policies.csv',index=False)
    fig,axs=plt.subplots(2,2,figsize=(8.2,5.8),layout='constrained')
    for reg,col in zip(['fixed','online','dual'],['#666666','#177e89','#b6404f']):
        for L,style in [(572.,'-'),(878.,'--')]:
            s=sorted([r for r in select('referral',premium=15.,demand=L) if r['regime']==reg],key=lambda r:r['model']['d2'])
            xx=[r['model']['d2'] for r in s]
            for ax,key,scale,title in zip(axs.flat,['objective','access','mu_on','p_on'],[1,100,100/972,1],
                ['Penalized objective per day','Strategic access (%)','Virtual capacity (%)','Virtual price']):
                ax.plot(xx,[r['result'][key]*scale for r in s],style,color=col,marker='o',ms=3,label=regime[reg]+f', {L:g}')
                ax.set(xlabel=r'In-person-to-virtual probability $\delta_2$',ylabel=title);ax.grid(alpha=.15)
    axs[0,0].legend(ncol=2,frameon=False);save(fig,'referral')
    fig,ax=plt.subplots(figsize=(6.3,4.3),layout='constrained')
    s=select('map');xs=sorted({r['model']['d1'] for r in s});ys=sorted({r['model']['d2'] for r in s})
    names=['B','V','F','BV','BF','VF','BVF'];colors=['#eeeeee','#3d89aa','#d5a148','#9acbdb','#efd399','#9a82b4','#516755']
    z=np.empty((len(ys),len(xs)))
    for r in s:z[ys.index(r['model']['d2']),xs.index(r['model']['d1'])]=names.index(r['result']['region'])
    mesh=ax.pcolormesh(xs,ys,z,shading='nearest',cmap=ListedColormap(colors),norm=BoundaryNorm(np.arange(-.5,7.5),7))
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):ax.text(x,max(y,.025),names[int(z[j,i])],ha='center',va='center',fontsize=8)
    xx=np.linspace(.08,.46,100);ax.plot(xx,xx/(1-xx),'k--',lw=1)
    ax.set(xlabel=r'Virtual-to-in-person probability $\delta_1$',ylabel=r'In-person-to-virtual probability $\delta_2$',ylim=(0,.82))
    save(fig,'region_map')
    # Actual numerical interpretation is derived from named rows.
    def find(L,s,d,reg='dual'):
        return next(r['result'] for r in rows if r['group']=='referral' and r['model']['demand']==L and r['model']['premium']==s and r['model']['d2']==d and r['regime']==reg)
    p0,p1=find(572,15,.1),find(878,15,.1)
    write('referral_text.tex',f'At $\\delta_2=0.1$ and $s=15$, the F2 policy uses support {p0["region"]} at $\\Lambda=572$ and {p1["region"]} at $\\Lambda=878$. '
          f'The corresponding access rates are {100*p0["access"]:.1f}\\% and {100*p1["access"]:.1f}\\%. '
          'Figure~\\ref{fig:referral} shows how changing in-person-to-virtual follow-up alone alters prices, capacity, and entry. '
          'The two-dimensional map in Figure~\\ref{fig:map} separates the workload threshold from the provider\'s preferred support: a workload ordering is not itself an objective-optimal switching rule. '
          'Full-grid outcomes, including zero entry under high fixed tariffs, remain in the comparison.\n')
    tr=[]
    for reg in regime:
        allr=[r for r in transfers if r['regime']==reg];valid=[r for r in allr if r['loss'] is not None]
        vals=[r['loss'] for r in valid]
        av=[r['loss'] for r in aware if r['regime']==reg]
        tr.append([regime[reg],str(len(allr)),str(len(allr)-len(valid)),number(np.median(vals)) if vals else '--',number(max(vals)) if vals else '--',number(np.median(av)) if av else '--',number(max(av)) if av else '--'])
    write('transfer_table.tex',table('Performance losses when simplified-model policies are implemented in the full staged system. The workload-matched benchmark preserves exogenous follow-up workload while omitting strategic in-person-to-virtual follow-up. Losses are absolute penalized-objective units.','tab:transfer','lrrrrrr',
        ['Regime','Cases','Omitted infeasible','Omitted median','Omitted max.','Adjusted median','Adjusted max.'],tr,'scriptsize'))
    pt=[]
    pricing_cases=[('A',572,15,0),('B',878,30,.1)]
    for case,L,s,d in pricing_cases:
        for reg in ['fixed','online','dual']:
            r=find(L,s,d,reg)
            capacity=f'({r["mu_on"]:.3f}, {r["mu_off"]:.3f})' if min(r['mu_on'],r['mu_off'])<.01 else f'({r["mu_on"]:.2f}, {r["mu_off"]:.2f})'
            pt.append([case,regime[reg],f'({r["p_on"]:.2f}, {r["p_off"]:.2f})',capacity,number(r['objective']),f'{100*r["access"]:.1f}'])
    write('pricing_table.tex',table('State-dependent value of nested pricing flexibility. Scenario A has $\\Lambda=572$, $s=15$, and $\\delta_2=0$; Scenario B has $\\Lambda=878$, $s=30$, and $\\delta_2=0.1$. Both use $\\delta_1=0.35$, $E=30$, and $M=972$.','tab:pricing','clllrr',
        ['Scenario','Regime','$(p_v,p_f)$','$(m_v,m_f)$','$J$','Access (\\%)'],pt,'scriptsize'))
    elasticity=[]
    for eta in [0,.01,.03]:
        x=next(r for r in rows if r['group']=='elasticity' and r['model']['elasticity']==eta)
        r=x['result']
        elasticity.append([f'{eta:.2f}',f'{r["required_actual"]:.2f}',f'{r["p_on"]:.2f}',f'{r["p_off"]:.2f}',f'{r["mu_on"]:.2f}',f'{r["mu_off"]:.2f}',number(r['objective']),f'{100*r["access"]:.2f}'])
    write('elasticity_table.tex',table('Price-responsive exogenous in-person demand under F2, with $\\Lambda=878$, $E_0=30$, $s=15$, $\\delta_1=0.35$, $\\delta_2=0.4$, and $M=972$.','tab:elasticity','rrrrrrrr',
        ['$\\eta$','$E(p)$','$p_v$','$p_f$','$m_v$','$m_f$','$J$','Access (\\%)'],elasticity,'scriptsize'))
    heterogeneity=[]
    for de in [0.,.1,.4,.7]:
        x=next(r for r in rows if r['group']=='heterogeneity' and r['model']['dE']==de)
        r=x['result']
        heterogeneity.append([f'{de:.1f}',number(r['objective']),f'{100*r["access"]:.2f}',
            f'{r["mu_on"]:.2f}',f'{r["p_on"]:.2f}',f'{r["p_off"]:.2f}'])
    write('deltaE_table.tex',table('Exogenous-patient follow-up sensitivity','tab:deltaE','rrrrrr',
        ['$\\delta_E$','$J$','Access (\\%)','$m_v$','$p_v$','$p_f$'],heterogeneity,'scriptsize'))
    fig,axs=plt.subplots(1,3,figsize=(8.5,2.8),layout='constrained')
    for variant,col,label in [('total','#b6404f','Fixed total capacity'),('residual','#177e89','Fixed aggregate residual capacity'),('no_income','#666666','Exogenous receipts removed')]:
        s=sorted([r for r in rows if r['group']=='required' and r['regime']=='dual' and r['label'].startswith(variant+'_')],key=lambda r:r['model']['required'])
        for ax,key,scale,title in zip(axs,['objective','access','p_off'],[1,100,1],['Penalized objective','Strategic access (%)','In-person price']):
            ax.plot([r['model']['required'] for r in s],[r['result'][key]*scale for r in s],marker='o',ms=3,color=col,label=label)
            ax.set(xlabel='Exogenous in-person arrivals per day',ylabel=title);ax.grid(alpha=.15)
    axs[0].legend(frameon=False,fontsize=7);save(fig,'required')
    r0=next(r['result'] for r in rows if r['group']=='required' and r['label']=='total_E0' and r['regime']=='dual')
    r90=next(r['result'] for r in rows if r['group']=='required' and r['label']=='total_E90' and r['regime']=='dual')
    a0,a90=[next(r['result'] for r in rows if r['group']=='required' and r['label']==f'residual_E{e}' and r['regime']=='dual') for e in [0,90]]
    write('required_text.tex',f'At fixed total capacity, increasing exogenous in-person arrivals from 0 to 90 changes strategic access from {100*r0["access"]:.2f}\\% to {100*r90["access"]:.2f}\\%, '
        f'while the objective changes from {number(r0["objective"])} to {number(r90["objective"])}. '
        f'When total capacity is adjusted to hold aggregate capacity net of exogenous workload fixed, access changes from {100*a0["access"]:.2f}\\% to {100*a90["access"]:.2f}\\%. '
        'This comparison does not hold residual capacity fixed within each service pool. '
        'The receipt-exclusion curve in Figure~\\ref{fig:required} retains physical load and removes exogenous-patient receipts only from the provider objective. '
        'Its monetary level is therefore not directly comparable to the other curves as a common accounting measure. Its policy changes identify the direct receipt incentive.\n')
    referral_staff=json.loads((STAFFING/'staffing_referral_transfers.json').read_text())
    referral_staff_by_label={z['label']:z for z in referral_staff}
    st=[]
    for x in staff:
        m=x['model'];r=x['full'];t=x['transferred']
        label=f'L{m["demand"]:g}_d{m["d2"]:g}_staged'
        st.append([f'{m["demand"]:g}',f'{m["d2"]:.1f}',f'{r["decision"]:.0f}',f'{r["p_on"]:.2f}',f'{r["p_off"]:.2f}',number(r['objective']),f'{100*r["access"]:.1f}',number(x['loss']) if x['loss'] is not None else 'Infeasible',number(referral_staff_by_label[label]['loss'])])
    write('staffing_table.tex',table('Multi-server policies, with $K=18$, $\\nu=54$, and $s=15$. Rate loss evaluates a rounded continuous-capacity policy. Referral loss evaluates a staffing policy that preserves exogenous follow-up workload but omits strategic in-person-to-virtual follow-up.','tab:staffing','rrrrrrrrr',
        ['$\\Lambda$','$\\delta_2$','$k_v$','$p_v$','$p_f$','$J$','Access (\\%)','Rate loss','Referral loss'],st,'scriptsize'))
    worst=max(staff,key=lambda x:x['loss'] if x['loss'] is not None else -1)
    write('staffing_text.tex',f'Table~\\ref{{tab:staffing}} shows that equal aggregate rate capacity does not make the two technologies interchangeable. '
        f'The largest observed implementation loss in these six comparisons is {number(worst["loss"])} objective units, at $\\Lambda={worst["model"]["demand"]:g}$ and $\\delta_2={worst["model"]["d2"]:.1f}$. '
        'The integer-staffing solution accounts for both parallel service and price-induced participation. '
        'A second comparison keeps the multi-server capacity representation in both planning models. The simpler model retains actual exogenous follow-up workload and omits only strategic in-person-to-virtual follow-up. '
        f'Its six transferred policies are stable, with objective losses ranging from {number(min(z["loss"] for z in referral_staff))} to {number(max(z["loss"] for z in referral_staff))}. '
        'Thus, the decision value of accounting for strategic follow-up also appears with parallel clinicians. The staffing decision incorporates the congestion function of the deployed capacity representation.\n')
    fig,ax=plt.subplots(figsize=(5.2,3.0),layout='constrained')
    for variant,col,label in [('one_referral','#888888','Truncated'),('staged','#177e89','Staged'),('recurrent','#b6404f','Recurrent')]:
        s=sorted([r for r in rows if r['group']=='pathway' and r['label'].startswith(variant+'_')],key=lambda r:r['model']['d2'])
        ax.plot([r['model']['d2'] for r in s],[100*r['result']['access'] for r in s],marker='o',color=col,label=label)
    ax.set(xlabel=r'$\delta_2$',ylabel='Strategic access (%)');ax.legend(frameon=False)
    save(fig,'pathway_comparison')
    fig,ax=plt.subplots(figsize=(5.2,3.0),layout='constrained')
    axr=ax.twinx()
    for copay,col in [(1.,'#177e89'),(.3,'#b6404f')]:
        s=sorted(select('payment',copay=copay),key=lambda r:r['model']['followup_fee'])
        xx=[r['model']['followup_fee'] for r in s]
        ax.plot(xx,[r['result']['objective'] for r in s],marker='o',color=col,label=rf'$\alpha={copay:g}$')
        axr.plot(xx,[100*r['result']['access'] for r in s],ls='--',color=col)
    ax.set(xlabel=r'Billing fraction for additional encounters $\theta$',ylabel='Objective (solid)')
    axr.set_ylabel('Access (%) (dashed)');ax.legend(frameon=False,loc='center')
    save(fig,'payment_arrangements')
    p= {r['label']:r['result'] for r in rows if r['group']=='pathway'}
    write('pathway_text.tex',f'At $\\delta_2=0.4$, strategic access is {100*p["one_referral_d0.4"]["access"]:.2f}\\% in the truncated-pathway system, '
        f'{100*p["staged_d0.4"]["access"]:.2f}\\% in the staged system, and {100*p["recurrent_d0.4"]["access"]:.2f}\\% in the recurrent system. '
        'The truncated-pathway and staged continuous-baseline policies use the regional characterization, whereas the recurrent policy is selected numerically under its distinct workload matrix. '
        'The payment comparison is reported separately in Supplemental Figure~\\ref{fig:payment-arrangements}.\n')
    sims={s['name']:s for s in sim};chosen=[sims[k] for k in ['moderate','high','prescribed_VF','prescribed_BVF'] if k in sims]
    sr=[]
    for s in chosen:
        r=s['policy'];stats=s['statistics'];j=stats['objective'];we=stats['episode_E']
        sr.append([case_name(s['name']),r['region'],number(r['objective']),f'{number(j["mean"])} $\\pm$ {number(j["halfwidth"])}',
                   f'{480*r["wait_E"]:.3f}',f'{480*we["mean"]:.3f} $\\pm$ {480*we["halfwidth"]:.3f}'])
    write('simulation_table.tex',table('Baseline DES validation: 10,000 independent replications per policy. Intervals are 95\\% half-widths; episode delays are in minutes. Mixed-support cases are constructed benchmarks for validating the event logic.','tab:des','llrrrr',
        ['Case','Support','Analytic $J$','Simulated $J$','Analytic $W_E$','Simulated $W_E$'],sr,'scriptsize'))
    errs=[abs(s['statistics']['objective']['mean']-s['policy']['objective'])/max(1,abs(s['policy']['objective']))*100 for s in chosen]
    write('simulation_text.tex',f'The largest absolute relative difference between the analytic objective and the simulation mean in Table~\\ref{{tab:des}} is {max(errs):.3f}\\%. '
          'Provider-receipt agreement checks encounter accounting. The independently recorded episode and encounter delays provide the congestion check. '
          'Supplemental Material~D reports all class-specific delay estimates and utility deviations, including virtual-entry checks for unused initial channels.\n')
    # Supplement tables are longtable so that every configured outcome remains visible.
    lines=[r'\begin{longtable}{p{1.8cm}p{3.35cm}lrrrr}',r'\caption{Full configured results. Capacity is a virtual-service rate for continuous models and a clinician count for staffing.}\label{tab:all}\\',
           r'\toprule Group & Case & Regime & $\Lambda$ & $J$ & Access (\%) & Capacity\\\midrule\endfirsthead',
           r'\toprule Group & Case & Regime & $\Lambda$ & $J$ & Access (\%) & Capacity\\\midrule\endhead']
    for row in rows:
        r=row['result'];m=row['model']
        lines.append(' & '.join([row['group'],row['label'].replace('_',' '),regime[row['regime']],f'{m["demand"]:.0f}',number(r['objective']),f'{100*r["access"]:.1f}',f'{r["decision"]:.1f}'])+r'\\')
    lines.extend([r'\bottomrule\end{longtable}']);write('full_table.tex','\n'.join(lines))
    # Definitions and class-specific checks follow the order used in Supplemental D.2.
    case_order=['moderate','high','prescribed_VF','prescribed_BVF',
                'mixed','mixed_high','staffing','long_window']
    sim=sorted(sim,key=lambda s:case_order.index(s['name']))
    case_sources={
        'moderate':'Selected baseline policy',
        'high':'Selected baseline policy',
        'prescribed_VF':'Prescribed mixed-route policy',
        'prescribed_BVF':'Prescribed mixed-route policy',
        'mixed':'Selected staged policy',
        'mixed_high':'Selected staged policy',
        'staffing':'Explicit staffing policy',
        'long_window':'Constructed BVF policy',
    }
    case_rows=[]
    for s in sim:
        m=s['model']
        setting=(f'{int(s["warmup"])} warm-up, {int(s["measurement"])} measurement days'
                 if s['name']=='long_window' else
                 rf'$\Lambda={m["demand"]:g}$, $s={m["premium"]:g}$, $\delta_2={m["d2"]:.1f}$'
                 +(', $K=18$' if s['name']=='staffing' else ''))
        case_rows.append([case_name(s['name']),rf'$\mathsf{{{s["policy"]["region"]}}}$',
                          case_sources[s['name']],setting])
    write('validation_cases.tex',table('Simulation validation cases','tab:validation-cases','llll',
        ['Case','Support','Policy source','Key setting'],case_rows,'scriptsize'))
    details=[];utility_rows=[]
    for s in sim:
        stats=s['statistics'];r=s['policy'];m=s['model']
        for key,theory in [('episode_on','wait_initial_on'),('episode_off','wait_initial_off'),('episode_E','wait_E'),
                           ('encounter_on_first','wait_on'),('encounter_on_followup','wait_on'),('encounter_off_first','wait_off'),('encounter_off_followup','wait_off')]:
            stt=stats[key]
            if stt:details.append([case_name(s['name']),DELAY_LABELS[key],f'{480*r[theory]:.4f}',f'{480*stt["mean"]:.4f}',f'{480*stt["halfwidth"]:.4f}'])
        data=np.load(SIMULATION/f'des_{s["name"]}_replications.npz')['values']
        from model import Model
        qm,em,hm,bem,vm=Model(**m).matrices()
        uv=vm-m['copay']*hm.T@np.array([r['p_on'],r['p_off']])
        for j,origin in enumerate(['virtual','in-person']):
            vals=uv[j]-data[:,20:22]@(np.array([m['cost_on'],m['cost_off']])*qm[:,j])
            half=student_t.ppf(.975,len(vals)-1)*np.std(vals,ddof=1)/np.sqrt(len(vals))
            utility_rows.append([case_name(s['name']),origin,f'{r["utility_on" if j==0 else "utility_off"]:.5f}',f'{np.mean(vals):.5f}',f'{half:.5f}'])
    def longtable(headers,lines,label,caption,cols):
        caption=short_caption(caption)
        heading='\\toprule '+' & '.join(headers)+r'\\\midrule'
        return '\\begin{longtable}{'+cols+'}\n\\caption{'+caption+'}\\label{'+label+'}\\\\\n'+heading+r'\endfirsthead'+'\n'+heading+r'\endhead'+'\n'+'\n'.join(' & '.join(r)+r'\\' for r in lines)+'\n\\bottomrule\\end{longtable}\n'
    write('delay_details.tex',longtable(['Case','Class','Analytical','DES mean','95\\% half-width'],details,
        'tab:delaydetails','Encounter and episode delay validation','llrrr'))
    write('utility_details.tex',longtable(['Case','Initial option','Analytical','DES estimate','95\\% half-width'],utility_rows,
        'tab:utilities','Patient utility validation','llrrr'))
    transfer_detail=[];transfer_export=[]
    for kind,items in [('Basic omission',transfers),('Workload-matched omission',aware)]:
        for reg in regime:
            cases=[z for z in items if z['regime']==reg];valid=[z for z in cases if z['transferred'] is not None]
            if valid:
                dj=np.median([z['loss'] for z in valid])
                da=100*np.median([z['full']['access']-z['transferred']['access'] for z in valid])
                dw=480*np.median([z['transferred']['wait_E']-z['full']['wait_E'] for z in valid])
                nums=[number(dj),f'{da:.2f}',number(dw,2)]
            else:nums=['--','--','--']
            transfer_detail.append([kind,regime[reg],str(len(cases)-len(valid))+'/'+str(len(cases)),*nums])
        for z in items:
            d=dict(benchmark=kind,label=z['label'],regime=regime[z['regime']],loss=z['loss'])
            d.update({'full_'+k:v for k,v in z['full'].items()})
            d.update({'transferred_'+k:v for k,v in (z['transferred'] or {}).items()})
            transfer_export.append(d)
    pd.DataFrame(transfer_export).to_csv(OUT/'policy_transfers.csv',index=False)
    write('transfer_details.tex',table('Implementation performance of follow-up-omission policies',
        'tab:transferdetails','llrrrr',
        ['Planning model','Regime','Infeasible','$\\Delta J$','$\\Delta A$','$\\Delta W_E$'],transfer_detail,'scriptsize'))
    reserves=read('reserve_sensitivity.json')
    rr=[]
    for z in reserves:
        m=z['model'];f=z['full'];t=z['transferred']
        rr.append([f'{z["required_slack"]:.0f}',f'{m["demand"]:.0f}',f'{m["d2"]:.1f}',number(f['objective']),number(t['objective']),number(z['loss']),f'{100*f["access"]:.1f}',f'{100*t["access"]:.1f}',number(480*t['wait_E'],2)])
    write('reserve_table.tex',table('Common capacity buffers in both F2 planning models, with $s=15$. Policies are recomputed for each buffer $\\varepsilon$. Subscripts F and S denote the full-model policy and the simplified-model policy implemented in the full system. Delay is in minutes.','tab:reserve','rrrrrrrrr',
        ['$\\varepsilon$','$\\Lambda$','$\\delta_2$','$J_F$','$J_S$','Loss','$A_F$ (\\%)','$A_S$ (\\%)','$W_{E,S}$'],rr,'scriptsize').replace('[ht]','[H]'))
    write('reserve_text.tex',f'A further twelve paired comparisons impose common capacity buffers of 54 or 108 encounter-rate units above exogenous workload at each service pool and reoptimize both planning models under the same buffer. '
        f'All transfers remain stable, with objective losses from {number(min(z["loss"] for z in reserves))} to {number(max(z["loss"] for z in reserves))}. '
        'The largest exogenous-patient delay effects occur when a simplified-model policy operates close to the stability boundary, where the reciprocal queueing delay increases sharply. With common buffers of 54 or 108 rate units, every implemented policy remains stable and the objective losses persist, so the decision differences are not explained solely by proximity to instability. Supplemental Material~C reports these comparisons.\n')
    if (STATE/'state_validation.json').exists():
        state=json.loads((STATE/'state_validation.json').read_text());lines=[]
        for s in state:
            r=s['policy'];st=s['statistics']
            lines.append([case_name(s['name']),f'{r["p_on"]:.2f}',f'{r["p_off"]:.2f}',
                f'{number(st["objective"]["mean"])} $\\pm$ {number(st["objective"]["halfwidth"])}',f'{100*st["access"]["mean"]:.2f}'])
        text=table('State-observing routing under alternative policies. All final comparisons use 10,000 replications.','tab:state','lrrrr',
            ['Policy','$p_v$','$p_f$','DES objective','Access (\\%)'],lines,'small')
        base=state[0];other=state[1]
        text+='The base and maximum-in-person-price policies have the same stationary objective and capacity, but different price allocations. '
        text+=f'Under the state-observing rule their access rates are {100*base["statistics"]["access"]["mean"]:.2f}\\% and {100*other["statistics"]["access"]["mean"]:.2f}\\%, respectively. '
        text+='The selected policy is the best observed in the specified 27-point neighborhood, evaluated with a fresh random stream. '
        base_values=np.load(STATE/f'des_{base["name"]}_replications.npz')['values']
        paired=[]
        for s in state[1:]:
            values=np.load(STATE/f'des_{s["name"]}_replications.npz')['values']
            delta=values[:,[1,2]]-base_values[:,[1,2]]
            mean=delta.mean(axis=0);half=student_t.ppf(.975,len(delta)-1)*delta.std(axis=0,ddof=1)/np.sqrt(len(delta))
            paired.append(dict(policy=s['name'],objective_difference=float(mean[0]),objective_halfwidth=float(half[0]),access_difference_pp=float(mean[1]*100),access_halfwidth_pp=float(half[1]*100)))
        (STATE/'state_paired.json').write_text(json.dumps(paired,indent=2))
        text+=f'Relative to the base policy, the selected policy increases the simulated objective by {number(paired[-1]["objective_difference"])} $\\pm$ {number(paired[-1]["objective_halfwidth"])} and access by {paired[-1]["access_difference_pp"]:.2f} $\\pm$ {paired[-1]["access_halfwidth_pp"]:.2f} percentage points (paired 95\\% intervals).\n'
        write('information_text.tex',text)
    else:write('information_text.tex','% Results generated after final state-observing evaluations.\n')
    # General audit, including nested feasible sets and negative transfer losses.
    for L in [572,878]:
        for s in [15,30]:
            for d in [0,.1,.2,.35,.5,.65,.8]:
                jj=[find(L,s,d,reg)['objective'] for reg in regime]
                assert jj[1]>=jj[0]-.01 and jj[2]>=jj[1]-.01,(L,s,d,jj)
    assert max(r['result']['residual'] for r in rows)<1e-5
    assert min(t['loss'] for t in transfers if t['loss'] is not None)>-.01
    if aware:assert min(t['loss'] for t in aware)>-.01
    print('Generated numerical exhibits; nesting, equilibrium residuals, and transfer signs checked.')

if __name__=='__main__':build()
