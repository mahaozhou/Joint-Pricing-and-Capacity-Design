"""Keep exhibit assets free of narrative and use short captions.

Applied before compilation so regenerated result tables keep the same convention.
Context, units, and interpretations belong to the manuscript or supplement.
Never move an old caption into prose or insert table notes.
"""
import re
from paths import TABLES
TITLES={
 'Clinical stages.':'Care pathway',
 'Independent changes in':'Effects of downstream virtual follow-up',
 'Optimal continuous-baseline F2':'Equilibrium regions',
 'Effects of exogenous in-person demand':'Effects of required in-person demand',
 'Care-pathway and payment effects':'Care pathways and payment arrangements',
 'Support conditions for':'Equilibrium support conditions',
 'Regional solvability of':'Regional solution methods',
 'Reference scenario values':'Reference parameters',
 'State-observing routing under':'Policies under observed congestion',
 'Baseline DES validation:':'Simulation validation',
 'Utility checks using':'Patient utility validation',
 'State-dependent value of':'Value of pricing flexibility',
 'Performance losses when':'Losses from simplified referral planning',
 'Full configured results.':'Detailed computational results',
 'Encounter and episode delays':'Encounter and episode delays',
 'Multi-server policies,':'Clinician allocation and policy performance',
 'Common capacity buffers':'Capacity buffer sensitivity',
 'Price-responsive exogenous':'Price-responsive in-person demand',
 'Performance of simplified-model':'Performance of simplified referral policies',
 'Utilization of every':'Service pool utilization',
 'Nested pricing regimes under':'Pricing flexibility with parallel clinicians',
 'Reoptimized multi-server':'Clinician allocation and policy performance',
}

def short_caption(caption):
    return next((v for k,v in TITLES.items() if caption.startswith(k)),caption)

def clean_asset(text):
    """Keep complete float environments only; narrative is author-maintained."""
    floats=list(re.finditer(r'\\begin\{(figure|table|longtable)\}[\s\S]*?\\end\{\1\}',text))
    if not floats:
        return text
    return '\n\n'.join(m.group(0) for m in floats)+'\n'

def shorten():
    changed=[]
    for p in sorted(TABLES.glob('*.tex')):
        s=p.read_text(encoding='utf-8');edits=[]
        for match in re.finditer(r'\\caption\{',s):
            a=match.end();i=a;depth=1
            while depth:
                if s[i]=='{':depth+=1
                elif s[i]=='}':depth-=1
                i+=1
            caption=s[a:i-1]
            title=short_caption(caption)
            if caption==title:continue
            edits.append((a,i-1,title))
            changed.append((p.name,title))
        original=s
        for a,b,t in sorted(edits,reverse=True):s=s[:a]+t+s[b:]
        s=clean_asset(s)
        if s!=original:
            p.write_text(s,encoding='utf-8')
    return changed

if __name__=='__main__':
    for row in shorten():print(*row,sep=': ')
