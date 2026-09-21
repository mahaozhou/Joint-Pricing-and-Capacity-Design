"""Generate the targeted AE exhibits from saved, independently optimized splits."""
from pathlib import Path
import json
from model import Model, evaluate, policy
from shorten_captions import short_caption

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/revision/many_server_audit'

def table(caption,label,columns,header,rows):
    caption=short_caption(caption)
    return '\n'.join([r'\begin{table}[ht]',r'\centering\footnotesize',
        r'\caption{'+caption+r'}\label{'+label+'}',
        r'\begin{tabular}{@{}'+columns+r'@{}}\toprule',header+r'\\\midrule',
        *[' & '.join(r)+r'\\' for r in rows],r'\bottomrule\end{tabular}',r'\end{table}'])+'\n'

def build():
    rows=json.loads((OUT/'selected.json').read_text())
    original=json.loads((ROOT/'results/revision/staffing_transfers.json').read_text())
    referral=json.loads((ROOT/'results/revision/staffing_referral_transfers.json').read_text())
    transfers=[];sr=[];ur=[];pr=[]
    regimes={'fixed':'F0','online':'F1','dual':'F2'}
    for row in rows:
        r=row['result'];m=Model(**row['model'])
        ur.append([row['label'].replace('_',r'\_'),regimes[row['regime']],str(r['k_v']),str(r['k_f']),
            f"{r['load_v']:.2f}",f"{r['load_f']:.2f}",f"{100*r['rho_v']:.2f}",f"{100*r['rho_f']:.2f}"])
        if row['label'].startswith('L'):
            old=next(x for x in original if x['label']==row['label']+'_staged')
            simple=next(x for x in referral if x['label']==row['label']+'_staged')
            c=old['continuous'];k=max(1,min(17,int(c['mu_on']/m.nu+.5)))
            implemented=evaluate(m,(k,c['p_on'],c['p_off']))
            transferred=evaluate(m,policy(simple['selected']))
            assert implemented is not None and transferred is not None
            loss=r['objective']-implemented['objective'];sloss=r['objective']-transferred['objective']
            transfers.append(dict(label=row['label'],model=row['model'],full=r,
                continuous=c,implemented=implemented,implementation_loss=loss,
                simplified_selected=simple['selected'],simplified_implemented=transferred,referral_loss=sloss,
                provenance='Full policy rerun in this audit; archived continuous and simplified policies reevaluated without changing their decisions.'))
            sr.append([f'{m.demand:g}',f'{m.d2:.1f}',f"({r['k_v']},{r['k_f']})",f"{r['p_on']:.2f}",f"{r['p_off']:.2f}",f"{r['objective']:,.1f}",f"{100*r['access']:.1f}",f'{loss:,.1f}',f'{sloss:,.1f}'])
        else:
            pr.append([row['label'],regimes[row['regime']],f"({r['k_v']},{r['k_f']})",f"{r['p_on']:.2f}",f"{r['p_off']:.2f}",f"{r['objective']:,.1f}",f"{100*r['access']:.1f}"])
    (OUT/'transfers.json').write_text(json.dumps(transfers,indent=2))
    (OUT/'staffing_table.tex').write_text(table(
        'Reoptimized multi-server policies for $K=18$, $\\nu=54$, $s=15$, and F2. Implementation loss retains continuous-policy prices and rounds its capacity to clinicians. Referral loss implements the workload-matched simplified policy in the full multi-server system; both losses are differences in $J$.',
        'tab:staffing','rrcrrrrrr',r'$\Lambda$ & $\delta_2$ & $(k_v,k_f)$ & $p_v$ & $p_f$ & $J$ & Access (\%) & Impl. loss & Ref. loss',sr))
    (OUT/'utilization_table.tex').write_text(table(
        'Utilization of every selected multi-server policy in the representative design. Labels L572 and L878 denote demand, and d denotes $\\delta_2$; these six cases use $s=15$ and F2. A and B use the representative pricing scenarios in the main text. Loads are encounters per clinic-day; utilization is in percent.',
        'tab:staff-utilization','llrrrrrr',r'Case & Regime & $k_v$ & $k_f$ & $\ell_v$ & $\ell_f$ & $\rho_v$ (\%) & $\rho_f$ (\%)',ur))
    (OUT/'pricing_table.tex').write_text(table(
        'Nested pricing regimes under explicit multi-server staffing. Scenario A: $\\Lambda=572$, $s=15$, $\\delta_2=0$. Scenario B: $\\Lambda=878$, $s=30$, $\\delta_2=0.1$. Both use $\\delta_1=0.35$, $E=30$, $K=18$, and $\\nu=54$. All 17 splits are separately evaluated under each regime.',
        'tab:staff-pricing','llcrrrr',r'Scenario & Regime & $(k_v,k_f)$ & $p_v$ & $p_f$ & $J$ & Access (\%)',pr))
    # The following assertions guard the interpretations in the article and response.
    get=lambda label,reg:next(x['result'] for x in rows if x['label']==label and x['regime']==reg)
    assert [get(f'L878_d{d}','dual')['k_v'] for d in ('0.1','0.4','0.7')]==[2,13,13]
    assert [get(f'L572_d{d}','dual')['k_v'] for d in ('0.1','0.4','0.7')]==[13,13,13]
    assert get('A','fixed')['objective']<get('A','online')['objective']<get('A','dual')['objective']
    assert abs(get('B','fixed')['objective']-get('B','online')['objective'])<1e-3
    assert get('B','online')['objective']<get('B','dual')['objective']
    assert min(t['referral_loss'] for t in transfers)>0
    (OUT/'main_findings.tex').write_text('% Narrative is maintained in the manuscript and supplement.\n',encoding='utf-8')
    print('Generated audit tables and checked every narrative ordering.')

if __name__=='__main__':build()
