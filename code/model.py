"""Experiment inputs and the C++ kernel interface."""
from dataclasses import dataclass, asdict
from pathlib import Path
from ctypes import CDLL, c_int, c_double, POINTER
from itertools import product
import math
import os
import subprocess
import time

import numpy as np


@dataclass(frozen=True)
class Scenario:
    demand: float = 572
    required: float = 30
    a: float = .35
    b: float = .2
    d: float = .25
    theta: float = .5
    alpha: float = 1
    V: float = 50
    s: float = 15
    T: float = 20
    cv: float = 112
    cf: float = 168
    penalty: float = 10
    nu_v: float = 54
    nu_f: float = 54
    K: int = 18
    path: str = 'S'
    gamma: float = 0
    receipt: int = 1
    elasticity: float = 0
    cap_v: float = 60
    cap_f: float = 80
    fixed_v: float = 60
    fixed_f: float = 40
    regime: str = 'F2'

    @property
    def box(self):
        if self.regime == 'F0':
            return self.fixed_v, self.fixed_v, self.fixed_f, self.fixed_f
        if self.regime == 'F1':
            return 0., self.cap_v, self.fixed_f, self.fixed_f
        return 0., self.cap_v, 0., self.cap_f

    @property
    def visits(self):
        b = 0 if self.path == 'N' else self.b
        gamma = self.gamma if self.path == 'R' else 0
        z = 1 - gamma*self.a*b
        ze = 1 - gamma*self.a*self.d
        shift = 0 if self.path in ('N', 'O') else self.a*b/z
        Q = np.array([[1 + shift, b/z], [self.a/z, 1/z]])
        H = np.array([[1 + self.theta*shift, self.theta*b/z], [self.a/z, 1/z]])
        e = np.array([self.d/ze, 1/ze])
        he = np.array([self.theta*self.d/ze, 1/ze])
        value = np.array([self.V + self.a*self.s - self.T*self.a/z,
                          self.V + self.s - self.T/z])
        return Q, H, e, he, value

    def encode(self):
        return np.array([self.demand, self.required, self.a, self.b, self.d, self.theta, self.alpha,
                         self.V, self.s, self.T, self.cv, self.cf, self.penalty, self.nu_v,
                         self.nu_f, self.K, 'NOSR'.index(self.path), self.gamma, self.receipt,
                         self.elasticity, *self.box, self.fixed_f], dtype=np.float64)


HERE = Path(__file__).resolve().parent
KEYS = ('J G x_v x_f ell_v ell_f W_v W_f E U_v U_f p_v p_f '
        'lower_bound point_upper upper_bound wardrop_residual support').split()
SUPPORT = ['exit', 'virtual', 'in-person', 'virtual+exit', 'in-person+exit',
           'both', 'both+exit']


def shared_library(name):
    return HERE / (name + (".dll" if os.name == "nt" else ".so"))


def build_shared(name, *options):
    platform_flags = ["-static"] if os.name == "nt" else ["-fPIC"]
    subprocess.run([os.environ.get("CXX", "g++"), "-std=c++17", "-O3", *options,
                    *platform_flags, "-shared", str(HERE / f"{name}.cpp"),
                    "-o", str(shared_library(name))], check=True)


def build():
    build_shared("kernel", "-ffp-contract=off")


def library():
    dll = CDLL(str(shared_library("kernel")))
    ptr = POINTER(c_double)
    dll.optimize.argtypes = ptr, c_int, c_double, c_int, ptr, ptr
    dll.optimize.restype = c_int
    dll.evaluate.argtypes = ptr, c_int, c_double, c_double, ptr
    dll.evaluate.restype = c_int
    return dll


_DLL = None


def _kernel():
    global _DLL
    if _DLL is None:
        _DLL = library()
    return _DLL


def _pointer(arr):
    return arr.ctypes.data_as(POINTER(c_double))


def decoded(a, i, sc, target=.1):
    out = {k: float(x) for k,x in zip(KEYS, a)}
    out['k_v'], out['k_f'] = i, sc.K-i
    out['gap'] = out['upper_bound'] - out['lower_bound']
    out['relative_gap'] = out['gap']/max(1., abs(out['J']))
    out['stable'] = bool(out['ell_v'] < i*sc.nu_v and out['ell_f'] < (sc.K-i)*sc.nu_f)
    out['support'] = SUPPORT[int(out['support'])] if out['support'] >= 0 else 'unknown'
    eps = 2e-6
    out['entry'] = ('both' if out['x_v'] > eps and out['x_f'] > eps else
                    'virtual' if out['x_v'] > eps else 'in-person' if out['x_f'] > eps else 'exit')
    out['access'] = (out['x_v'] + out['x_f'])/sc.demand
    out['fup_share_v'] = (out['ell_v']-out['x_v'])/out['ell_v'] if out['ell_v'] else None
    e = sc.visits[2]
    out['mandatory_minutes'] = 480*(e[0]*out['W_v'] + e[1]*out['W_f'])
    out['status'] = 'epsilon_certified' if out['gap'] <= target else 'bounded_candidate'
    checks = [(sc.box[0], sc.box[1], out['p_v']), (sc.box[2], sc.box[3],out['p_f'])]
    out['price_bindings'] = '|'.join('fixed' if l == h else 'lower' if abs(l-x)<1.e-7 else
                                     'upper' if abs(h-x)<1.e-7 else 'interior' for l,h,x in checks)
    out['binding_participation'] = bool(max(out['U_v'], out['U_f']) <= 2e-6)
    out['binding_entry_indifference'] = abs(out['U_v']-out['U_f']) <= 2e-6
    return out


def optimum(sc, fixed=0, target=.1, budget=1600):
    start = time.perf_counter()
    data = sc.encode()
    out, splits = np.zeros(22), np.zeros((sc.K-1,20))
    status = _kernel().optimize(_pointer(data),fixed,target,budget,_pointer(out),_pointer(splits))
    if status != 1:
        raise RuntimeError(f"Optimization status {status}: {sc} fixed={fixed}")
    k = int(out[18])
    best = decoded(out, k, sc,target)
    best.update(boxes=int(out[19]),evaluations=int(out[20]),admissible_splits=int(out[21]),
                elapsed=time.perf_counter()-start, target=target, node_limit=budget,
                certificate_type='outward_price_boxes')
    rivals = []
    for i,z in enumerate(splits,1):
        if not z[19]:
            continue
        p = decoded(z,i,sc,target)
        p['boxes'] = int(z[18])
        rivals.append(p)
    best['possible_optimal_k'] = [p['k_v'] for p in rivals if p['upper_bound']>=best['lower_bound']]
    best['unique_optimal_k'] = (len(best['possible_optimal_k'])==1 and
                               best['possible_optimal_k'][0]==k)
    return best,rivals


def evaluate(sc,k,pv,pf):
    out=np.zeros(18)
    status=_kernel().evaluate(_pointer(sc.encode()),k,pv,pf,_pointer(out))
    if status==0:
        return None
    if status!=1:
        raise RuntimeError(f"Evaluation status {status}: {sc} {k} {pv} {pf}")
    return decoded(out,k,sc)


def nominal_w(load,k,rate):
    if load >= k*rate:
        return math.inf
    z=load/rate
    term=total=1.
    for j in range(1,k):
        term *= z/j
        total += term
    gap=k*rate-load
    tail=term*z*rate/gap
    return 1/rate+(tail/(total+tail))/gap


def full_flow(sc, k, entry, price=None):
    p=np.array([sc.box[1],sc.box[3]] if price is None else price)
    x=np.array([sc.demand,0.] if entry=='virtual' else [0.,sc.demand])
    Q,H,e,he,v=sc.visits
    E=sc.required*math.exp(-sc.elasticity*max(0.,sc.alpha*(he@p-sc.fixed_f)))
    ell=Q@x+E*e
    w=np.array([nominal_w(ell[0],k,sc.nu_v),nominal_w(ell[1],sc.K-k,sc.nu_f)])
    if not np.all(np.isfinite(w)):
        return None
    u=v-sc.alpha*(p@H)-np.array([sc.cv,sc.cf])*w@Q
    return dict(k_v=k,k_f=sc.K-k,p_v=float(p[0]),p_f=float(p[1]),x_v=x[0],x_f=x[1],ell_v=ell[0],
                ell_f=ell[1],W_v=w[0],W_f=w[1],U_v=u[0],U_f=u[1],
                J=float(p@(H@x+sc.receipt*E*he)),entry=entry, E=E)
