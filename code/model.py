"""Stage-aware outpatient network; all rates use an eight-hour clinic-day unit.

The fixed-policy Wardrop solver enumerates supports, not equilibria: under the
conditions documented in the manuscript the aggregate equilibrium is unique.
Numba accelerates scalar bracketing; no fitted surrogate is used.
"""
from dataclasses import dataclass, asdict, replace
import numpy as np
from numba import njit
from scipy.optimize import differential_evolution, minimize


@dataclass(frozen=True)
class Model:
    demand: float = 572.
    required: float = 30.
    d1: float = .35
    d2: float = .10
    gamma: float = -1.  # -1: same offline follow-up probability d2
    dE: float = -1.
    value: float = 50.
    premium: float = 30.
    travel: float = 20.
    preference: float = 0.
    cost_on: float = 112.
    cost_off: float = 168.
    capacity: float = 972.
    cap_on: float = 60.
    cap_off: float = 80.
    penalty: float = 10.
    followup_fee: float = 1.
    copay: float = 1.
    revenue_E: float = 1.  # zero is an accounting counterfactual, not a payment policy
    elasticity: float = 0.
    recurrent: bool = False
    servers: int = 0
    nu: float = 54.

    def matrices(self):
        g = self.d2 if self.gamma < 0 else self.gamma
        de = self.d2 if self.dE < 0 else self.dE
        if self.recurrent:
            q = np.array([[1., self.d2], [self.d1, 1.]])/(1-self.d1*self.d2)
            e = np.array([de, 1.])/(1-self.d1*de)
        else:
            q = np.array([[1+self.d1*g, self.d2], [self.d1, 1.]])
            e = np.array([de, 1.])
        b = np.eye(2)+self.followup_fee*(q-np.eye(2))
        be = np.array([self.followup_fee*e[0], 1+self.followup_fee*(e[1]-1)])
        v = np.array([self.value+self.preference+self.d1*self.premium,
                      self.value+self.premium])-self.travel*q[1]
        return q, e, b, be, v


@njit(cache=True)
def delay(load, capacity, k):
    if capacity <= load or capacity <= 0:
        return 1.e12
    if k <= 1:
        return 1./(capacity-load)
    nu = capacity/k
    a = max(0., load)/nu
    term = 1.
    total = 1.
    for n in range(1, k):
        term *= a/n
        total += term
    term *= a/k
    tail = term/(1-load/capacity)
    pw = tail/(total+tail)
    return 1/nu+pw/(capacity-load)


@njit(cache=True)
def utility(x, q, e, capacities, ks, costs, a):
    loads = q@x+e
    w = np.array([delay(loads[0], capacities[0], ks[0]),
                  delay(loads[1], capacities[1], ks[1])])
    return a-q.T@(costs*w), w


@njit(cache=True)
def wardrop(L, q, e, capacities, ks, costs, a):
    """Returns (flows, delays, support code); -1 means compulsory load unstable."""
    empty = np.zeros(2)
    if np.min(capacities-e) <= 0:
        return empty, np.ones(2)*1.e12, -1
    tol = 2.e-7
    u, w = utility(empty, q, e, capacities, ks, costs, a)
    if max(u) <= tol:
        return empty, w, 0
    # Pure full-coverage supports.
    for j in range(2):
        x = np.zeros(2)
        x[j] = L
        if np.min(capacities-e-q@x) > 0:
            u, w = utility(x, q, e, capacities, ks, costs, a)
            if u[j] >= -tol and u[j] >= u[1-j]-tol:
                return x, w, 1+j
    # Single active channel and nonparticipation: strictly decreasing own utility.
    for j in range(2):
        hi = L
        for h in range(2):
            if q[h,j] > 0:
                hi = min(hi, (capacities[h]-e[h])/q[h,j]*(1-1.e-12))
        lo = 0.
        xx = np.zeros(2)
        u0, _ = utility(xx, q, e, capacities, ks, costs, a)
        xx[j] = hi
        uh, _ = utility(xx, q, e, capacities, ks, costs, a)
        if u0[j] >= 0 and uh[j] <= 0:
            for _ in range(52):
                mid = (lo+hi)/2
                xx[j] = mid
                um, _ = utility(xx, q, e, capacities, ks, costs, a)
                if um[j] > 0:
                    lo = mid
                else:
                    hi = mid
            xx[j] = (lo+hi)/2
            u, w = utility(xx, q, e, capacities, ks, costs, a)
            if u[1-j] <= tol:
                return xx, w, 3+j
    # Both channels, full coverage: difference strictly decreases along the edge.
    lo, hi = 0., L
    for h in range(2):
        slope = q[h,0]-q[h,1]
        rhs = capacities[h]-e[h]-q[h,1]*L
        if abs(slope) < 1.e-14:
            if rhs <= 0:
                hi = -1.
        elif slope > 0:
            hi = min(hi, rhs/slope-1.e-9)
        else:
            lo = max(lo, rhs/slope+1.e-9)
    if hi >= lo:
        xl = np.array([lo,L-lo]); xh = np.array([hi,L-hi])
        ul, _ = utility(xl,q,e,capacities,ks,costs,a)
        uh, _ = utility(xh,q,e,capacities,ks,costs,a)
        if ul[0]-ul[1] >= -tol and uh[0]-uh[1] <= tol:
            for _ in range(52):
                mid = (lo+hi)/2
                um, _ = utility(np.array([mid,L-mid]),q,e,capacities,ks,costs,a)
                if um[0] > um[1]: lo=mid
                else: hi=mid
            x = np.array([(lo+hi)/2,L-(lo+hi)/2])
            u,w=utility(x,q,e,capacities,ks,costs,a)
            if min(u) >= -tol:
                return x,w,5
    # Both channels and nonparticipation: invert target delays then encounter matrix.
    det = q[0,0]*q[1,1]-q[0,1]*q[1,0]
    target = np.array([(q[1,1]*a[0]-q[1,0]*a[1])/det/costs[0],
                       (-q[0,1]*a[0]+q[0,0]*a[1])/det/costs[1]])
    loads = np.zeros(2)
    ok = True
    for h in range(2):
        if target[h] < delay(0.,capacities[h],ks[h]):
            ok=False
            break
        if ks[h] <= 1:
            loads[h] = capacities[h]-1/target[h]
        else:
            lo,hi=0.,capacities[h]*(1-1.e-12)
            for _ in range(52):
                mid=(lo+hi)/2
                if delay(mid,capacities[h],ks[h]) < target[h]: lo=mid
                else: hi=mid
            loads[h]=(lo+hi)/2
    if ok:
        z=loads-e
        x=np.array([(q[1,1]*z[0]-q[0,1]*z[1])/det,
                    (-q[1,0]*z[0]+q[0,0]*z[1])/det])
        if min(x)>=-1.e-6 and sum(x)<=L+1.e-6:
            x=np.maximum(x,0.)
            u,w=utility(x,q,e,capacities,ks,costs,a)
            return x,w,6
    return empty,np.ones(2)*1.e12,-2


REGIONS = ['B','V','F','BV','BF','VF','BVF']

def evaluate(m, policy):
    mu, p0, p1 = map(float,policy)
    if m.servers:
        k0=int(round(mu)); ks=np.array([k0,m.servers-k0],dtype=np.int64)
        caps=ks*m.nu
    else:
        ks=np.ones(2,dtype=np.int64); caps=np.array([mu,m.capacity-mu])
    if min(ks)<1 or min(caps)<=0 or p0 < -1e-7 or p1 < -1e-7 or p0>m.cap_on+1e-7 or p1>m.cap_off+1e-7:
        return None
    q,e,b,be,v=m.matrices(); prices=np.array([p0,p1])
    # Price response is a transparent scenario, referenced to initial tariff 40.
    req=m.required*np.exp(-m.elasticity*max(0.,be@prices-40.))
    a=v-m.copay*b.T@prices
    x,w,r=wardrop(float(m.demand),q,e*req,caps,ks,np.array([m.cost_on,m.cost_off]),a)
    if r<0: return None
    gross=float(prices@(b@x+be*req))
    objective=gross-(1-m.revenue_E)*float(prices@be*req)-m.penalty*(m.demand-sum(x))
    u=a-q.T@(np.array([m.cost_on,m.cost_off])*w)
    maxu=max(0.,*u)
    residual=max([maxu-u[i] for i in range(2) if x[i]>1e-5]+([maxu] if sum(x)<m.demand-1e-5 else [0.]))
    return dict(mu_on=float(caps[0]),mu_off=float(caps[1]),decision=float(mu),p_on=p0,p_off=p1,
                flow_on=float(x[0]),flow_off=float(x[1]),required_actual=float(req),region=REGIONS[r],
                gross=gross,penalty=m.penalty*(m.demand-sum(x)),objective=objective,
                access=float(sum(x)/m.demand),wait_on=float(w[0]),wait_off=float(w[1]),
                wait_E=float(e@w),wait_initial_on=float(q[:,0]@w),wait_initial_off=float(q[:,1]@w),
                exposure=float((q@x+e*req)@w/(sum(x)+req)) if sum(x)+req>0 else 0.,
                utility_on=float(u[0]),utility_off=float(u[1]),residual=float(residual),
                referral_forward=float(m.d1*x[0]),referral_reverse=float(m.d2*(x[1]+req)+m.d1*(m.d2 if m.gamma<0 else m.gamma)*x[0]))


def optimize(m, regime='dual', seed=17, iterations=180, restarts=2, secondary=True, fixed_capacity=None, required_slack=1e-3):
    """Global-search heuristic with explicit equilibrium; no global certificate claimed."""
    all_solutions=[]
    counts=range(1,m.servers) if m.servers and fixed_capacity is None else [fixed_capacity]
    for count in counts:
        q,e,b,be,v=m.matrices()
        bounds=[]
        if count is None:
            bounds.append((max(required_slack,e[0]*m.required+required_slack),m.capacity-e[1]*m.required-required_slack))
        if regime!='fixed': bounds.append((0.,m.cap_on))
        if regime=='dual': bounds.append((0.,m.cap_off))
        def unpack(z):
            t=0
            cap=count
            if cap is None: cap=z[t]; t+=1
            pon=60.
            if regime!='fixed': pon=z[t]; t+=1
            poff=40.
            if regime=='dual': poff=z[t]
            return (cap,pon,poff)
        def objective(z):
            r=evaluate(m,unpack(z))
            return 1.e9 if r is None else -r['objective']
        if not bounds:
            rr=evaluate(m,unpack([]))
            if rr: all_solutions.append(rr)
            continue
        if any(hi<=lo for lo,hi in bounds): continue
        best=None
        restart_values=[]
        for restart in range(restarts):
            fit=differential_evolution(objective,bounds,seed=seed+restart*107+int(count or 0),
                maxiter=iterations,popsize=12,tol=2e-9,atol=1e-5,polish=True)
            if best is None or fit.fun<best.fun: best=fit
            restart_values.append(-float(fit.fun))
        rr=evaluate(m,unpack(best.x))
        if rr is None: continue
        jstar=rr['objective']
        if secondary:
            def exposure(z):
                r=evaluate(m,unpack(z)); return 1e6 if r is None else r['exposure']
            fit=minimize(exposure,best.x,method='SLSQP',bounds=bounds,
                constraints=[{'type':'ineq','fun':lambda z:-objective(z)-jstar+1e-4}],
                options={'maxiter':180,'ftol':1e-11})
            alt=evaluate(m,unpack(fit.x))
            if alt and alt['objective']>=jstar-1.1e-4 and alt['exposure']<=rr['exposure']:
                rr=alt
        rr['search_best']=jstar
        rr['search_spread']=max(restart_values)-min(restart_values)
        all_solutions.append(rr)
    if not all_solutions: raise ValueError('No stable required-load allocation')
    j=max(r['objective'] for r in all_solutions)
    best=min([r for r in all_solutions if r['objective']>=j-2e-4],key=lambda r:r['exposure'])
    best['regime']=regime
    return best


def policy(r): return (r['decision'],r['p_on'],r['p_off'])
