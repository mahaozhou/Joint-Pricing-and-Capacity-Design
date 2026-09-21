import numpy as np
from scipy.optimize import minimize
from model import Model, evaluate, optimize, policy, delay

def test_erlang_one_and_zero():
    assert abs(delay(30.,54.,1)-1/24)<1e-12
    assert abs(delay(0.,108.,2)-1/54)<1e-12

def test_stage_matrix_and_workload_threshold():
    m=Model(d1=.35,d2=.6)
    q,*_=m.matrices()
    assert abs(np.linalg.det(q)-1)<1e-12
    assert q[:,0].sum()<q[:,1].sum()
    q,*_=Model(d1=.35,d2=.4).matrices()
    assert q[:,0].sum()>q[:,1].sum()

def test_equilibrium_against_convex_potential():
    rng=np.random.default_rng(23)
    for _ in range(30):
        m=Model(d1=rng.uniform(.1,.7),d2=rng.uniform(0,.7),premium=rng.uniform(10,30))
        pp=(rng.uniform(180,650),rng.uniform(10,60),rng.uniform(20,75))
        r=evaluate(m,pp)
        assert r is not None
        q,e,b,be,v=m.matrices(); caps=np.array([pp[0],m.capacity-pp[0]])
        a=v-b.T@np.array(pp[1:]); c=np.array([m.cost_on,m.cost_off])
        def f(y):
            z=caps-e*m.required-q@(y*m.demand)
            if min(z)<=0: return 1e12
            return (-a@(y*m.demand)-c@np.log(z))/m.demand
        def jac(y):
            z=caps-e*m.required-q@(y*m.demand)
            return -a+q.T@(c/np.maximum(z,1e-10))
        sol=minimize(f,[0.,0.],jac=jac,method='SLSQP',bounds=[(0,1)]*2,
            constraints=[{'type':'ineq','fun':lambda y:1-sum(y)},
                         {'type':'ineq','fun':lambda y:(caps-e*m.required-q@(y*m.demand))/m.capacity-1e-9}],
            options={'ftol':1e-11,'maxiter':300})
        if not sol.success or np.max(np.abs(sol.x*m.demand-[r['flow_on'],r['flow_off']]))>.03:
            from scipy.optimize import LinearConstraint, Bounds
            sol=minimize(f,[1e-6,1e-6],jac=jac,method='trust-constr',
                hess=lambda y:m.demand*q.T@np.diag(c/(caps-e*m.required-q@(y*m.demand))**2)@q,
                bounds=Bounds([0,0],[1,1],keep_feasible=True),
                constraints=[LinearConstraint([[1,1]],-np.inf,1,keep_feasible=True),
                    LinearConstraint(q*m.demand,-np.inf,caps-e*m.required-1e-8,keep_feasible=True)],
                options={'gtol':1e-11,'maxiter':1000})
        assert sol.success
        assert np.max(np.abs(sol.x*m.demand-[r['flow_on'],r['flow_off']]))<.03
        assert r['residual']<1e-5
        assert abs(r['gross']-r['penalty']-r['objective'])<1e-6

def test_zero_required_and_oneway():
    m=Model(d2=0,required=0,gamma=0)
    r=evaluate(m,(250,40,40))
    assert r and r['residual']<1e-5

def test_old_benchmark_accounting():
    r=evaluate(Model(gamma=0),(134.63006566885622,60,40))
    assert abs(r['gross']-27692)<.01
    assert r['region']=='F'

def test_multiserver_equilibrium():
    m=Model(servers=18)
    for k in [2,5,9,14]:
        r=evaluate(m,(k,30,40))
        assert r and r['residual']<1e-5

def test_support_coverage_across_payment_and_pathway_variants():
    rng=np.random.default_rng(912)
    for i in range(500):
        m=Model(demand=float(rng.uniform(200,1200)),d1=float(rng.uniform(0,.9)),
                d2=float(rng.uniform(0,.9)),gamma=0. if i%3==0 else -1.,
                required=float(rng.uniform(0,40)),premium=float(rng.uniform(10,35)),
                followup_fee=float(rng.choice([0,.5,1])),copay=float(rng.choice([.3,1])),
                recurrent=i%4==0,servers=18 if i%5==0 else 0)
        q,e,*_=m.matrices()
        mu=float(rng.integers(3,15)) if m.servers else float(rng.uniform(max(150,e[0]*m.required+1),min(800,972-e[1]*m.required-1)))
        r=evaluate(m,(mu,float(rng.uniform(0,60)),float(rng.uniform(0,80))))
        if r is None:
            caps=np.array([mu*54,(18-mu)*54]) if m.servers else np.array([mu,972-mu])
            assert np.any(caps<=e*m.required)
        else:assert r['residual']<1e-5
