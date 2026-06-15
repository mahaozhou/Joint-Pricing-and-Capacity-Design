# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Callable, List
import os, math, statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import brentq, differential_evolution, least_squares, linprog


# =============================================================================
# Plot styles for price figures
# =============================================================================

CH4_PRICE_COLOR = 'tab:green'
CH4_PRICE_MARKER = '^'
CH5_PON_COLOR = 'tab:red'
CH5_POFF_COLOR = 'tab:purple'
CH5_PRICE_MARKER = 's'
PRICE_LINEWIDTH = 1.8
PRICE_MARKERSIZE = 6


# =============================================================================
# Utilities
# =============================================================================

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_and_show(fig, out_path: str):
    pdf_path = os.path.splitext(out_path)[0] + '.pdf'
    fig.savefig(pdf_path, bbox_inches="tight", format='pdf')
    plt.close(fig)


def golden_section_max(f: Callable[[float], float], a: float, b: float,
                       tol: float = 1e-6, maxit: int = 200) -> Tuple[float, float]:
    if b <= a:
        return a, f(a)
    phi = (1 + 5 ** 0.5) / 2
    rho = 1 - 1 / phi
    c = b - rho * (b - a)
    d = a + rho * (b - a)
    fc = f(c)
    fd = f(d)
    it = 0
    while (b - a) > tol and it < maxit:
        if fc < fd:
            a, c, fc = c, d, fd
            d = a + rho * (b - a)
            fd = f(d)
        else:
            b, d, fd = d, c, fc
            c = b - rho * (b - a)
            fc = f(c)
        it += 1
    x_star = (a + b) / 2
    f_star = f(x_star)
    return x_star, f_star


# =============================================================================
# Erlang-C (M/M/c); when c=1, total time = Wq + 1/mu
# =============================================================================

def erlang_c_P0(lam: float, mu_ps: float, c: int) -> float:
    if c <= 0 or mu_ps <= 0:
        return 0.0
    rho = lam / (c * mu_ps)
    if rho >= 1.0:
        return 0.0
    a = lam / mu_ps
    s = 1.0
    term = 1.0
    for k in range(1, c):
        term *= a / k
        s += term
    term *= a / c
    s += term / (1.0 - rho)
    return 1.0 / s


def erlang_c_wait(lam: float, mu_ps: float, c: int) -> float:
    if c <= 0 or mu_ps <= 0:
        return float('inf')
    rho = lam / (c * mu_ps)
    if rho >= 1.0:
        return float('inf')
    P0 = erlang_c_P0(lam, mu_ps, c)
    a = lam / mu_ps
    term = 1.0
    for k in range(1, c + 1):
        term *= a / k
    Pw = term * P0 / (1.0 - rho)
    Wq = Pw / (c * mu_ps - lam)
    return Wq + 1.0 / mu_ps


# =============================================================================
# Chapter 4 — fixed p_off (psi optional) + p_on cap
# =============================================================================

@dataclass
class Ch4Params:
    Lambda: float
    lambda_E: float
    delta1: float
    delta2: float
    V: float
    T: float
    s: float
    p_off: float
    mu_bar: float
    # --- split waiting costs ---
    Cw_on: float
    Cw_off: float
    # --- infection avoidance benefit (kept as baseline param; no sweep) ---
    alpha: float = 0.0
    # --- ---
    psi: float = 0.0
    r_balk: float = 0.0
    M_const: float = 0.0
    outer_grid: int = 301
    refine_topk: int = 5
    eps: float = 1e-9
    # --- NEW: p_on cap ---
    pbar_on: float = 60.0


@dataclass
class Ch4Result:
    region: str
    mu_on: float
    mu_off: float
    p_on: float
    p_off: float
    lambda_on: float
    lambda_off: float
    lambda_balk: float
    revenue: float
    details: Dict[str, Any]


# --- helpers ---

def D_const(d1: float, d2: float) -> float:
    return 1.0 - d1 * d2


def B_req_off_ch4(p: Ch4Params) -> float:
    D = D_const(p.delta1, p.delta2)
    # Y0^psi = ((1 - d2) V - d2 psi)/D
    Y0psi = ((1.0 - p.delta2) * p.V - p.delta2 * p.psi) / D
    return Y0psi - (p.T - p.s) - p.p_off


def revenue_ch4(p: Ch4Params, p_on: float, lam_on: float, lam_off: float) -> float:
    L, lE = p.Lambda, p.lambda_E
    d1, d2 = p.delta1, p.delta2
    base = ((lam_on + d2 * (lam_off + lE)) * p_on
            + (lam_off + lE + d1 * lam_on) * p.p_off
            - (L - lam_on - lam_off) * p.r_balk + p.M_const)
    return base + p.alpha * lam_on


def _cap_pon_nonneg(x: float, p: Ch4Params) -> float:
    return max(0.0, min(x, p.pbar_on))


# --- regions (with p_on cap applied) ---

def solve_ch4_region_B(mu_on: float, p: Ch4Params) -> Optional[Ch4Result]:
    mu_off = p.mu_bar - mu_on
    d_on = mu_on - p.delta2 * p.lambda_E
    d_off = mu_off - p.lambda_E
    if d_on <= p.eps or d_off <= p.eps:
        return None
    A = p.Cw_on / d_on
    B = p.Cw_off / d_off
    rhs_on = p.V + p.psi - A - p.delta1 * (p.T + p.p_off + B) + p.delta1 * p.s
    if p.delta2 > 0:
        rhs_off = (p.V - (p.T + p.p_off + B) + p.s) / p.delta2 - A
        p_on = max(rhs_on, rhs_off)
    else:
        if p.V - (p.T + p.p_off + B) + p.s > 1e-9:
            return None
        p_on = max(rhs_on, 0.0)
    p_on = _cap_pon_nonneg(p_on, p)
    rev = revenue_ch4(p, p_on, 0.0, 0.0)
    return Ch4Result("B", mu_on, mu_off, p_on, p.p_off, 0.0, 0.0, p.Lambda, rev, {"A": A, "B": B})


def solve_ch4_region_V(mu_on: float, p: Ch4Params) -> Optional[Ch4Result]:
    L, d1 = p.Lambda, p.delta1
    mu_off = p.mu_bar - mu_on
    if mu_on <= (L + p.delta2 * p.lambda_E) + p.eps:
        return None
    if mu_off <= (d1 * L + p.lambda_E) + p.eps:
        return None
    Breq = B_req_off_ch4(p)
    if Breq <= 0:
        return None
    HV = p.mu_bar - ((1.0 + d1) * L + (1.0 + p.delta2) * p.lambda_E)
    if HV <= 0:
        return None
    x_sq = HV / (1.0 + math.sqrt(d1))
    y_sq = math.sqrt(d1) * HV / (1.0 + math.sqrt(d1))
    y_cap = p.Cw_off / Breq
    if y_sq <= y_cap + p.eps:
        x, y = x_sq, y_sq
    else:
        y = y_cap
        x = HV - y
        if x <= 0:
            return None
    A = p.Cw_on / x; B = p.Cw_off / y
    p_on = (p.V + p.psi) - A - d1 * (p.T + p.p_off + B) + d1 * p.s
    p_on = _cap_pon_nonneg(p_on, p)
    lam_on, lam_off = L, 0.0
    U_off = p.V - (p.T + p.p_off + B) - p.delta2 * (p_on + A) + p.s
    if U_off > 1e-9:
        return None
    rev = revenue_ch4(p, p_on, lam_on, lam_off)
    return Ch4Result("V", mu_on, mu_off, p_on, p.p_off, lam_on, lam_off, 0.0, rev, {"A": A, "B": B})


def solve_ch4_region_F(mu_on: float, p: Ch4Params) -> Optional[Ch4Result]:
    L, d2 = p.Lambda, p.delta2
    mu_off = p.mu_bar - mu_on
    if mu_off <= (L + p.lambda_E) + p.eps or mu_on <= d2 * (L + p.lambda_E) + p.eps:
        return None
    Breq = B_req_off_ch4(p)
    if Breq <= 0:
        return None
    HF = p.mu_bar - ((1.0 + d2) * (L + p.lambda_E))
    if HF <= 0:
        return None
    x_sq = math.sqrt(d2) * HF / (1.0 + math.sqrt(d2))
    y_sq = HF / (1.0 + math.sqrt(d2))
    y_min = p.Cw_off / Breq
    if y_sq >= y_min - p.eps:
        x, y = x_sq, y_sq
    else:
        y = y_min
        x = HF - y
        if x <= 0:
            return None
    A = p.Cw_on / x; B = p.Cw_off / y
    if d2 == 0:
        p_on = (p.V + p.psi) - A - p.delta1 * (p.T + p.p_off + B) + p.delta1 * p.s
    else:
        KF = p.V - (p.T - p.s)
        p_on = (KF - p.p_off - B) / d2 - A
    p_on = _cap_pon_nonneg(p_on, p)
    lam_on, lam_off = 0.0, L
    U_on = p.V + p.psi - p_on - A - p.delta1 * (p.T + p.p_off + B) + p.delta1 * p.s
    U_off = p.V - (p.T + p.p_off + B) - p.delta2 * (p_on + A) + p.s
    if U_on - U_off > 1e-9 or U_off < -1e-9:
        return None
    rev = revenue_ch4(p, p_on, lam_on, lam_off)
    return Ch4Result("F", mu_on, mu_off, p_on, p.p_off, lam_on, lam_off, 0.0, rev, {"A": A, "B": B})


def solve_ch4_region_BV(mu_on: float, p: Ch4Params) -> Optional[Ch4Result]:
    L, d1, d2 = p.Lambda, p.delta1, p.delta2
    mu_off = p.mu_bar - mu_on
    if mu_on <= p.eps or mu_off <= p.eps:
        return None

    def obj(lam_on: float) -> float:
        if not (0.0 < lam_on < L):
            return -math.inf
        if mu_on - (lam_on + d2 * p.lambda_E) <= p.eps:
            return -math.inf
        if mu_off - (d1 * lam_on + p.lambda_E) <= p.eps:
            return -math.inf
        A = p.Cw_on / (mu_on - (lam_on + d2 * p.lambda_E))
        B = p.Cw_off / (mu_off - (d1 * lam_on + p.lambda_E))
        p_on = (p.V + p.psi) - A - d1 * (p.T + p.p_off + B) + d1 * p.s
        p_on = _cap_pon_nonneg(p_on, p)
        U_off = p.V - (p.T + p.p_off + B) - d2 * (p_on + A) + p.s
        if U_off > 1e-9:
            return -math.inf
        return revenue_ch4(p, p_on, lam_on, 0.0)

    upper = min(L - p.eps, mu_on - d2 * p.lambda_E - p.eps)
    if d1 > 0:
        upper = min(upper, (mu_off - p.lambda_E) / d1 - p.eps)
    lower = p.eps
    if upper <= lower:
        return None
    lam_on_star, rev_star = golden_section_max(obj, lower, upper)
    if not math.isfinite(rev_star):
        return None
    lam_on = lam_on_star
    A = p.Cw_on / (mu_on - (lam_on + d2 * p.lambda_E))
    B = p.Cw_off / (mu_off - (d1 * lam_on + p.lambda_E))
    p_on = (p.V + p.psi) - A - d1 * (p.T + p.p_off + B) + d1 * p.s
    p_on = _cap_pon_nonneg(p_on, p)
    lam_balk = max(0.0, L - lam_on)
    return Ch4Result("BV", mu_on, mu_off, p_on, p.p_off, lam_on, 0.0, lam_balk, rev_star, {"A": A, "B": B})


def solve_ch4_region_BF(mu_on: float, p: Ch4Params) -> Optional[Ch4Result]:
    L, d1, d2 = p.Lambda, p.delta1, p.delta2
    mu_off = p.mu_bar - mu_on
    if mu_on <= p.eps or mu_off <= p.eps:
        return None

    def obj(lam_off: float) -> float:
        if not (0.0 < lam_off < L):
            return -math.inf
        if mu_on - d2 * (lam_off + p.lambda_E) <= p.eps:
            return -math.inf
        if mu_off - (lam_off + p.lambda_E) <= p.eps:
            return -math.inf
        A = p.Cw_on / (mu_on - d2 * (lam_off + p.lambda_E))
        B = p.Cw_off / (mu_off - (lam_off + p.lambda_E))
        if d2 == 0:
            p_on = (p.V + p.psi) - A - d1 * (p.T + p.p_off + B) + d1 * p.s
        else:
            KF = p.V - (p.T - p.s)
            p_on = (KF - p.p_off - B) / d2 - A
        p_on = _cap_pon_nonneg(p_on, p)
        U_on = p.V + p.psi - p_on - A - d1 * (p.T + p.p_off + B) + d1 * p.s
        if U_on > 1e-9:
            return -math.inf
        return revenue_ch4(p, p_on, 0.0, lam_off)

    upper = min(L - p.eps, (mu_off - p.lambda_E) - p.eps)
    if d2 > 0:
        upper = min(upper, (mu_on - p.delta2 * p.lambda_E) / d2 - p.eps)
    lower = p.eps
    if upper <= lower:
        return None
    lam_off_star, rev_star = golden_section_max(obj, lower, upper)
    if not math.isfinite(rev_star):
        return None
    lam_off = lam_off_star
    A = p.Cw_on / (mu_on - d2 * (lam_off + p.lambda_E))
    B = p.Cw_off / (mu_off - (lam_off + p.lambda_E))
    if d2 == 0:
        p_on = (p.V + p.psi) - A - d1 * (p.T + p.p_off + B) + d1 * p.s
    else:
        KF = p.V - (p.T - p.s)
        p_on = (KF - p.p_off - B) / d2 - A
    p_on = _cap_pon_nonneg(p_on, p)
    lam_balk = max(0.0, L - lam_off)
    return Ch4Result("BF", mu_on, mu_off, p_on, p.p_off, 0.0, lam_off, lam_balk, rev_star, {"A": A, "B": B})


def solve_ch4_region_VF(mu_on: float, p: Ch4Params) -> Optional[Ch4Result]:
    L, d1 = p.Lambda, p.delta1
    mu_off = p.mu_bar - mu_on
    if mu_on <= p.eps or mu_off <= p.eps:
        return None
    Breq = B_req_off_ch4(p)
    if Breq <= 0 or abs(1.0 - d1) < p.eps:
        return None
    lam_on = (p.Cw_off / Breq - (mu_off - (L + p.lambda_E))) / (1.0 - d1)
    lam_off = L - lam_on
    if lam_on <= p.eps or lam_off <= p.eps:
        return None
    d_on = mu_on - (lam_on + p.delta2 * (lam_off + p.lambda_E))
    if d_on <= p.eps:
        return None
    A = p.Cw_on / d_on; B = Breq
    X0psi = ((1.0 - d1) * p.V + p.psi) / D_const(d1, p.delta2)
    p_on = X0psi - A
    p_on = _cap_pon_nonneg(p_on, p)
    U_on = p.V + p.psi - (p_on + A) - d1 * (p.T + p.p_off + B) + d1 * p.s
    U_off = p.V - (p.T + p.p_off + B) - p.delta2 * (p_on + A) + p.s
    if abs(U_on) > 1e-5 or abs(U_off) > 1e-5:
        return None
    rev = revenue_ch4(p, p_on, lam_on, lam_off)
    return Ch4Result("VF", mu_on, mu_off, p_on, p.p_off, lam_on, lam_off, 0.0, rev, {"A": A, "B": B})


def solve_ch4_region_BVF(mu_on: float, p: Ch4Params) -> Optional[Ch4Result]:
    L, d1, d2 = p.Lambda, p.delta1, p.delta2
    mu_off = p.mu_bar - mu_on
    if mu_on <= p.eps or mu_off <= p.eps:
        return None
    Breq = B_req_off_ch4(p)
    if Breq <= 0:
        return None

    def lam_off_of(lam_on: float) -> float:
        return mu_off - p.lambda_E - d1 * lam_on - p.Cw_off / Breq

    D = 1.0 - d1 * d2
    A0 = mu_on - d2 * mu_off + (d2 * p.Cw_off / Breq)
    ub = float('inf')
    if d1 > 0:
        ub = min(ub, (mu_off - p.lambda_E - p.Cw_off / Breq) / d1 - p.eps)
    else:
        if lam_off_of(p.eps) <= p.eps:
            return None
    ub = min(ub, A0 / D - p.eps)
    ub = min(ub, (L - mu_off + p.lambda_E + p.Cw_off / Breq) / (1.0 - d1) - p.eps)
    lb = p.eps
    if not (ub > lb):
        return None
    X0psi = ((1.0 - d1) * p.V + p.psi) / D

    def obj(lam_on: float) -> float:
        lam_off = lam_off_of(lam_on)
        if lam_off <= 0.0 or lam_on <= 0.0 or lam_on + lam_off >= L:
            return -math.inf
        d_on = A0 - D * lam_on
        if d_on <= p.eps:
            return -math.inf
        A = p.Cw_on / d_on
        p_on = X0psi - A
        p_on = _cap_pon_nonneg(p_on, p)
        return revenue_ch4(p, p_on, lam_on, lam_off)

    lam_on_star, rev_star = golden_section_max(obj, lb, ub)
    if not math.isfinite(rev_star):
        return None
    lam_on = lam_on_star
    lam_off = lam_off_of(lam_on)
    d_on = A0 - D * lam_on
    if d_on <= p.eps:
        return None
    A = p.Cw_on / d_on
    p_on = X0psi - A
    p_on = _cap_pon_nonneg(p_on, p)
    lam_balk = max(0.0, L - lam_on - lam_off)
    return Ch4Result("BVF", mu_on, mu_off, p_on, p.p_off, lam_on, lam_off, lam_balk,
                     rev_star, {"A": A, "B": Breq})


def ch4_best_region_at_mu(mu_on: float, p: Ch4Params) -> Optional[Ch4Result]:
    best = None
    for solver in (solve_ch4_region_B, solve_ch4_region_V, solve_ch4_region_F,
                   solve_ch4_region_BV, solve_ch4_region_BF,
                   solve_ch4_region_VF, solve_ch4_region_BVF):
        try:
            r = solver(mu_on, p)
        except Exception:
            r = None
        if r is None or not math.isfinite(r.revenue):
            continue
        if best is None or r.revenue > best.revenue:
            best = r
    return best


def ch4_refine_mu(mu_seed: float, p: Ch4Params, width: float) -> Optional[Ch4Result]:
    a = max(p.eps, mu_seed - width)
    b = min(p.mu_bar - p.eps, mu_seed + width)
    if b <= a:
        return None

    def f(x: float) -> float:
        r = ch4_best_region_at_mu(x, p)
        return -math.inf if r is None else r.revenue

    mu_star, _ = golden_section_max(f, a, b, 1e-4)
    return ch4_best_region_at_mu(mu_star, p)


def ch4_solve_global(p: Ch4Params) -> Ch4Result:
    if not (0.0 <= p.delta1 < 1.0 and 0.0 <= p.delta2 < 1.0):
        raise ValueError("delta1, delta2 must lie in [0,1)")
    if p.Cw_on <= 0 or p.Cw_off <= 0 or p.mu_bar <= 0:
        raise ValueError("Cw_on>0, Cw_off>0 and mu_bar>0 required")
    grid = [p.eps + i * (p.mu_bar - 2 * p.eps) / (p.outer_grid - 1)
            for i in range(p.outer_grid)]
    best = None
    seeds: List[Tuple[float, float]] = []
    for mu_on in grid:
        r = ch4_best_region_at_mu(mu_on, p)
        if r is None:
            continue
        seeds.append((mu_on, r.revenue))
        if best is None or r.revenue > best.revenue:
            best = r
    if best is None:
        raise RuntimeError("No feasible region for Chapter 4 under given params.")
    seeds.sort(key=lambda t: t[1], reverse=True)
    width = max(1e-3, 0.05 * p.mu_bar)
    for mu_seed, _ in seeds[: p.refine_topk]:
        r = ch4_refine_mu(mu_seed, p, width)
        if r is not None and r.revenue > best.revenue:
            best = r
    return best


# =============================================================================
# Chapter 5 — joint pricing (offline price capped by pbar_off) with psi
# =============================================================================

@dataclass
class Ch5Params:
    Lambda: float
    lambda_E: float
    delta1: float
    delta2: float
    V: float
    T: float
    s: float
    mu_bar: float
    # --- price caps ---
    pbar_on: float = 60.0
    pbar_off: float = 80.0
    # --- waits ---
    Cw_on: float = 14.0
    Cw_off: float = 21.0
    # --- infection-avoidance (kept as baseline param; no sweep) ---
    alpha: float = 0.0
    # --- ---
    psi: float = 0.0
    r_balk: float = 0.0
    M_const: float = 0.0
    outer_grid: int = 301
    refine_topk: int = 5
    eps: float = 1e-9


@dataclass
class Ch5Result:
    region: str
    mu_on: float
    mu_off: float
    p_on: float
    p_off: float
    lambda_on: float
    lambda_off: float
    lambda_balk: float
    revenue: float
    details: Dict[str, Any]


# --- helpers ---

def D(d1: float, d2: float) -> float:
    return 1.0 - d1 * d2


def bundles_X0_Y0(p: Ch5Params) -> Tuple[float, float]:
    return ((1.0 - p.delta1) * p.V + p.psi) / D(p.delta1, p.delta2), \
           ((1.0 - p.delta2) * p.V - p.delta2 * p.psi) / D(p.delta1, p.delta2)


def revenue_ch5(p: Ch5Params, p_on: float, p_off: float, lam_on: float, lam_off: float) -> float:
    Lon = lam_on + p.delta2 * (lam_off + p.lambda_E)
    Loff = lam_off + p.lambda_E + p.delta1 * lam_on
    base = Lon * p_on + Loff * p_off - (p.Lambda - lam_on - lam_off) * p.r_balk + p.M_const
    return base + p.alpha * lam_on


def waits_ch5(p: Ch5Params, mu_on: float, mu_off: float,
              lam_on: float, lam_off: float) -> Tuple[float, float]:
    d_on = mu_on - (lam_on + p.delta2 * (lam_off + p.lambda_E))
    d_off = mu_off - (lam_off + p.lambda_E + p.delta1 * lam_on)
    if d_on <= p.eps or d_off <= p.eps:
        return math.inf, math.inf
    return p.Cw_on / d_on, p.Cw_off / d_off


def prices_slack(p: Ch5Params, A: float, B: float) -> Tuple[float, float]:
    X0, Y0 = bundles_X0_Y0(p)
    p_on = X0 - A
    p_off = Y0 - (p.T - p.s) - B
    if p_on > p.pbar_on:
        p_on = p.pbar_on
    if p_on < 0:
        p_on = 0.0
    return p_on, p_off


def prices_V_BV_cap_binding(p: Ch5Params, A: float, B: float) -> Tuple[float, float]:
    Ybar = p.pbar_off + B + (p.T - p.s)
    X = (p.V + p.psi) - p.delta1 * Ybar
    p_on = X - A
    if p_on > p.pbar_on:
        p_on = p.pbar_on
    if p_on < 0:
        p_on = 0.0
    return p_on, p.pbar_off


def prices_F_BF_cap_choice(p: Ch5Params, A: float, B: float, cap_binds: bool) -> Tuple[float, float]:
    X0, Y0 = bundles_X0_Y0(p)
    if cap_binds:
        Y = p.pbar_off + B + (p.T - p.s)
    else:
        Y = Y0
    if p.delta2 > 0:
        X = (p.V - Y) / p.delta2
    else:
        X = max(p.V - p.delta1 * Y, 0.0)
    p_on = X - A
    if p_on > p.pbar_on:
        p_on = p.pbar_on
    if p_on < 0:
        p_on = 0.0
    p_off = Y - (p.T - p.s) - B if not cap_binds else p.pbar_off
    return p_on, p_off


# --- regions ---

def solve_ch5_region_B(mu_on: float, p: Ch5Params) -> Optional[Ch5Result]:
    mu_off = p.mu_bar - mu_on
    if mu_on <= p.eps or mu_off <= p.eps:
        return None
    lam_on = 0.0; lam_off = 0.0
    A = p.Cw_on / mu_on; B = p.Cw_off / mu_off
    p_off = min(p.pbar_off, p.pbar_off)
    rhs_on = p.V + p.psi - A - p.delta1 * (p_off + B + (p.T - p.s))
    if p.delta2 > 0:
        rhs_off = (p.V - (p.T - p.s) - p_off - B) / p.delta2 - A
        p_on = max(rhs_on, rhs_off)
    else:
        if p.V - (p.T - p.s) - p_off - B > 1e-9:
            return None
        p_on = max(rhs_on, 0.0)
    p_on = min(max(p_on, 0.0), p.pbar_on)
    rev = revenue_ch5(p, p_on, p_off, lam_on, lam_off)
    return Ch5Result("B", mu_on, mu_off, p_on, p_off, 0.0, 0.0, p.Lambda, rev, {"A": A, "B": B})


def solve_ch5_region_V(mu_on: float, p: Ch5Params) -> Optional[Ch5Result]:
    L = p.Lambda
    mu_off = p.mu_bar - mu_on
    lam_on, lam_off = L, 0.0
    A, B = waits_ch5(p, mu_on, mu_off, lam_on, lam_off)
    if not math.isfinite(A):
        return None
    X0, Y0 = bundles_X0_Y0(p)
    p_off_lin = Y0 - (p.T - p.s) - B
    if p_off_lin <= p.pbar_off + 1e-12:
        p_on, p_off = prices_slack(p, A, B)
    else:
        Bmin = Y0 - (p.T - p.s) - p.pbar_off
        if B + 1e-12 < Bmin:
            return None
        p_on, p_off = prices_V_BV_cap_binding(p, A, B)
    X = p_on + A; Y = p_off + B + (p.T - p.s)
    U_off = p.V - Y - p.delta2 * X
    if U_off > 1e-9:
        return None
    rev = revenue_ch5(p, p_on, p_off, lam_on, lam_off)
    return Ch5Result("V", mu_on, mu_off, p_on, p_off, lam_on, lam_off, 0.0, rev, {"A": A, "B": B})


def solve_ch5_region_F(mu_on: float, p: Ch5Params) -> Optional[Ch5Result]:
    L = p.Lambda
    mu_off = p.mu_bar - mu_on
    lam_on, lam_off = 0.0, L
    A, B = waits_ch5(p, mu_on, mu_off, lam_on, lam_off)
    if not math.isfinite(A):
        return None
    X0, Y0 = bundles_X0_Y0(p)
    p_off_lin = (Y0 - (p.T - p.s) - B)
    cap_binds = p_off_lin > p.pbar_off + 1e-12
    if cap_binds:
        Bmax = Y0 - (p.T - p.s) - p.pbar_off
        if B - 1e-12 > Bmax:
            return None
    p_on, p_off = prices_F_BF_cap_choice(p, A, B, cap_binds)
    X = p_on + A; Y = p_off + B + (p.T - p.s)
    U_on = p.V + p.psi - X - p.delta1 * Y
    U_off = p.V - Y - p.delta2 * X
    if U_on - U_off > 1e-9 or U_off < -1e-9:
        return None
    rev = revenue_ch5(p, p_on, p_off, lam_on, lam_off)
    return Ch5Result("F", mu_on, mu_off, p_on, p_off, lam_on, lam_off, 0.0, rev, {"A": A, "B": B})


def solve_ch5_region_BV(mu_on: float, p: Ch5Params) -> Optional[Ch5Result]:
    L, d1 = p.Lambda, p.delta1
    mu_off = p.mu_bar - mu_on
    if mu_on <= p.eps or mu_off <= p.eps:
        return None
    X0, Y0 = bundles_X0_Y0(p)

    def phi(lam_on: float) -> float:
        if not (0.0 < lam_on < L):
            return -math.inf
        A, B = waits_ch5(p, mu_on, mu_off, lam_on, 0.0)
        if not math.isfinite(A):
            return -math.inf
        p_off_lin = Y0 - (p.T - p.s) - B
        if p_off_lin <= p.pbar_off + 1e-12:
            p_on, p_off = X0 - A, p_off_lin
            p_on = min(max(p_on, 0.0), p.pbar_on)
        else:
            Bmin = Y0 - (p.T - p.s) - p.pbar_off
            if B + 1e-12 < Bmin:
                return -math.inf
            p_on, p_off = prices_V_BV_cap_binding(p, A, B)
        X = p_on + A; Y = p_off + B + (p.T - p.s)
        U_off = p.V - Y - p.delta2 * X
        if U_off > 1e-9:
            return -math.inf
        return revenue_ch5(p, p_on, p_off, lam_on, 0.0)

    upper = min(L - p.eps, mu_on - p.delta2 * p.lambda_E - p.eps)
    if d1 > 0:
        upper = min(upper, (mu_off - p.lambda_E) / d1 - p.eps)
    lower = p.eps
    if upper <= lower:
        return None
    lam_on_star, rev_star = golden_section_max(phi, lower, upper)
    if not math.isfinite(rev_star):
        return None
    lam_on = lam_on_star
    A, B = waits_ch5(p, mu_on, mu_off, lam_on, 0.0)
    p_off_lin = Y0 - (p.T - p.s) - B
    if p_off_lin <= p.pbar_off + 1e-12:
        p_on, p_off = X0 - A, p_off_lin
        p_on = min(max(p_on, 0.0), p.pbar_on)
    else:
        p_on, p_off = prices_V_BV_cap_binding(p, A, B)
    return Ch5Result("BV", mu_on, mu_off, p_on, p_off, lam_on, 0.0, L - lam_on,
                     rev_star, {"A": A, "B": B})


def solve_ch5_region_BF(mu_on: float, p: Ch5Params) -> Optional[Ch5Result]:
    L, d2 = p.Lambda, p.delta2
    mu_off = p.mu_bar - mu_on
    if mu_on <= p.eps or mu_off <= p.eps:
        return None
    X0, Y0 = bundles_X0_Y0(p)

    def phi(lam_off: float) -> float:
        if not (0.0 < lam_off < L):
            return -math.inf
        A, B = waits_ch5(p, mu_on, mu_off, 0.0, lam_off)
        if not math.isfinite(A):
            return -math.inf
        p_off_lin = Y0 - (p.T - p.s) - B
        cap_binds = (p_off_lin > p.pbar_off + 1e-12)
        if cap_binds:
            Bmax = Y0 - (p.T - p.s) - p.pbar_off
            if B - 1e-12 > Bmax:
                return -math.inf
        p_on, p_off = prices_F_BF_cap_choice(p, A, B, cap_binds)
        X = p_on + A; Y = p_off + B + (p.T - p.s)
        U_on = p.V + p.psi - X - p.delta1 * Y
        if U_on > 1e-9:
            return -math.inf
        return revenue_ch5(p, p_on, p_off, 0.0, lam_off)

    upper = min(L - p.eps, (mu_off - p.lambda_E) - p.eps)
    if d2 > 0:
        upper = min(upper, (mu_on - p.delta2 * p.lambda_E) / d2 - p.eps)
    lower = p.eps
    if upper <= lower:
        return None
    lam_off_star, rev_star = golden_section_max(phi, lower, upper)
    if not math.isfinite(rev_star):
        return None
    lam_off = lam_off_star
    A, B = waits_ch5(p, mu_on, mu_off, 0.0, lam_off)
    p_off_lin = Y0 - (p.T - p.s) - B
    cap_binds = (p_off_lin > p.pbar_off + 1e-12)
    p_on, p_off = prices_F_BF_cap_choice(p, A, B, cap_binds)
    return Ch5Result("BF", mu_on, mu_off, p_on, p_off, 0.0, lam_off, L - lam_off,
                     rev_star, {"A": A, "B": B})


def solve_ch5_region_VF(mu_on: float, p: Ch5Params) -> Optional[Ch5Result]:
    L = p.Lambda
    mu_off = p.mu_bar - mu_on
    X0, Y0 = bundles_X0_Y0(p)

    def phi(lam_off: float) -> float:
        if not (0.0 < lam_off < L):
            return -math.inf
        lam_on = L - lam_off
        A, B = waits_ch5(p, mu_on, mu_off, lam_on, lam_off)
        if not math.isfinite(A):
            return -math.inf
        p_off_lin = Y0 - (p.T - p.s) - B
        Bmin = Y0 - (p.T - p.s) - p.pbar_off
        if p_off_lin > p.pbar_off + 1e-12 and B + 1e-12 < Bmin:
            return -math.inf
        p_on, p_off = prices_slack(p, A, B)
        if p_off > p.pbar_off + 1e-12:
            return -math.inf
        return revenue_ch5(p, p_on, p_off, lam_on, lam_off)

    lam_off_star, rev_star = golden_section_max(phi, p.eps, L - p.eps)
    if not math.isfinite(rev_star):
        return None
    lam_off = lam_off_star; lam_on = L - lam_off
    A, B = waits_ch5(p, mu_on, mu_off, lam_on, lam_off)
    p_on, p_off = prices_slack(p, A, B)
    return Ch5Result("VF", mu_on, mu_off, p_on, p_off, lam_on, lam_off, 0.0,
                     rev_star, {"A": A, "B": B})


def solve_ch5_region_BVF(mu_on: float, p: Ch5Params) -> Optional[Ch5Result]:
    L = p.Lambda
    mu_off = p.mu_bar - mu_on
    if mu_on <= p.eps or mu_off <= p.eps:
        return None
    X0, Y0 = bundles_X0_Y0(p)
    Bmin = Y0 - (p.T - p.s) - p.pbar_off
    best: Optional[Ch5Result] = None
    if Bmin > 0 and abs(1.0 - p.delta1) > p.eps:
        def phi_cap(lam_on: float) -> float:
            if lam_on <= 0.0:
                return -math.inf
            lam_off = mu_off - p.lambda_E - p.delta1 * lam_on - p.Cw_off / Bmin
            if lam_off <= 0.0 or lam_on + lam_off >= L:
                return -math.inf
            A, B = waits_ch5(p, mu_on, mu_off, lam_on, lam_off)
            if not math.isfinite(A) or abs(B - Bmin) > 1e-6:
                return -math.inf
            p_on = X0 - A
            p_on = min(max(p_on, 0.0), p.pbar_on)
            p_off = p.pbar_off
            return revenue_ch5(p, p_on, p_off, lam_on, lam_off)

        upper = L - p.eps
        if p.delta1 > 0:
            upper = min(upper, (mu_off - p.lambda_E - p.Cw_off / Bmin) / p.delta1 - p.eps)
        lam_on_star, val = golden_section_max(phi_cap, p.eps, max(p.eps, upper))
        if math.isfinite(val):
            lam_on = lam_on_star
            lam_off = mu_off - p.lambda_E - p.delta1 * lam_on - p.Cw_off / Bmin
            A, B = waits_ch5(p, mu_on, mu_off, lam_on, lam_off)
            best = Ch5Result("BVF", mu_on, p.mu_bar - mu_on, min(X0 - A, p.pbar_on), p.pbar_off,
                             lam_on, lam_off, L - lam_on - lam_off, val, {"A": A, "B": B})

    def phi_slack_pair(lam_on: float, lam_off: float) -> float:
        if lam_on <= 0.0 or lam_off <= 0.0 or lam_on + lam_off >= L:
            return -math.inf
        A, B = waits_ch5(p, mu_on, mu_off, lam_on, lam_off)
        if not math.isfinite(A):
            return -math.inf
        p_on, p_off = prices_slack(p, A, B)
        if p_off > p.pbar_off + 1e-12:
            return -math.inf
        return revenue_ch5(p, p_on, p_off, lam_on, lam_off)

    def inner_max_off(lam_on: float) -> Tuple[float, float]:
        lo = p.eps; hi = max(p.eps, L - lam_on - p.eps)
        if hi <= lo:
            return lo, -math.inf
        def g(lam_off: float) -> float:
            return phi_slack_pair(lam_on, lam_off)
        return golden_section_max(g, lo, hi)

    def G(lam_on: float) -> float:
        _, val = inner_max_off(lam_on); return val

    lam_on_star, val = golden_section_max(G, p.eps, L - p.eps)
    if math.isfinite(val):
        lam_off_star, _ = inner_max_off(lam_on_star)
        lam_on = lam_on_star; lam_off = lam_off_star
        A, B = waits_ch5(p, mu_on, mu_off, lam_on, lam_off)
        p_on, p_off = prices_slack(p, A, B)
        cand = Ch5Result("BVF", mu_on, p.mu_bar - mu_on, p_on, p_off,
                         lam_on, lam_off, L - lam_on - lam_off, val, {"A": A, "B": B})
        if best is None or cand.revenue > best.revenue:
            best = cand
    return best


def ch5_best_region_at_mu(mu_on: float, p: Ch5Params) -> Optional[Ch5Result]:
    best = None
    for solver in (solve_ch5_region_B, solve_ch5_region_V, solve_ch5_region_F,
                   solve_ch5_region_BV, solve_ch5_region_BF,
                   solve_ch5_region_VF, solve_ch5_region_BVF):
        try:
            r = solver(mu_on, p)
        except Exception:
            r = None
        if r is None or not math.isfinite(r.revenue):
            continue
        if best is None or r.revenue > best.revenue:
            best = r
    return best


def ch5_refine_mu(mu_seed: float, p: Ch5Params, width: float) -> Optional[Ch5Result]:
    a = max(p.eps, mu_seed - width)
    b = min(p.mu_bar - p.eps, mu_seed + width)
    if b <= a:
        return None

    def f(x: float) -> float:
        r = ch5_best_region_at_mu(x, p)
        return -math.inf if r is None else r.revenue

    mu_star, _ = golden_section_max(f, a, b, 1e-4)
    return ch5_best_region_at_mu(mu_star, p)


def ch5_solve_global(p: Ch5Params) -> Ch5Result:
    if not (0.0 <= p.delta1 < 1.0 and 0.0 <= p.delta2 < 1.0):
        raise ValueError("delta1, delta2 must lie in [0,1)")
    if p.Cw_on <= 0 or p.Cw_off <= 0 or p.mu_bar <= 0:
        raise ValueError("Cw_on>0, Cw_off>0 and mu_bar>0 required")
    grid = [p.eps + i * (p.mu_bar - 2 * p.eps) / (p.outer_grid - 1)
            for i in range(p.outer_grid)]
    best = None
    seeds: List[Tuple[float, float]] = []
    for mu_on in grid:
        r = ch5_best_region_at_mu(mu_on, p)
        if r is None:
            continue
        seeds.append((mu_on, r.revenue))
        if best is None or r.revenue > best.revenue:
            best = r
    if best is None:
        raise RuntimeError("No feasible region for Chapter 5 under given params.")
    seeds.sort(key=lambda t: t[1], reverse=True)
    width = max(1e-3, 0.05 * p.mu_bar)
    for mu_seed, _ in seeds[: p.refine_topk]:
        r = ch5_refine_mu(mu_seed, p, width)
        if r is not None and r.revenue > best.revenue:
            best = r
    return best


# =============================================================================
# Publication runner: policy-space optimization with equilibrium verification
# =============================================================================
#
# The closed-form regional routines above are retained as useful analytical
# candidate generators.  The numerical study uses the verified policy-space
# runner below because a regional feasible set may be disconnected after price
# caps are imposed.  A single golden-section search across infeasible gaps can
# otherwise skip valid, high-revenue policies.

_regional_ch4_solve_global = ch4_solve_global
_regional_ch5_solve_global = ch5_solve_global


def _policy_waits(p, mu_on: float, mu_off: float,
                  lam_on: float, lam_off: float) -> Optional[Tuple[float, float]]:
    d_on = mu_on - (lam_on + p.delta2 * (lam_off + p.lambda_E))
    d_off = mu_off - (lam_off + p.lambda_E + p.delta1 * lam_on)
    if d_on <= p.eps or d_off <= p.eps:
        return None
    return p.Cw_on / d_on, p.Cw_off / d_off


def _policy_utilities(p, p_on: float, p_off: float,
                      A: float, B: float) -> Tuple[float, float]:
    u_on = (p.V + getattr(p, "psi", 0.0) - p_on - A
            - p.delta1 * (p.T + p_off + B) + p.delta1 * p.s)
    u_off = (p.V + p.s - p.T - p_off - B
             - p.delta2 * (p_on + A))
    return u_on, u_off


def _find_scalar_roots(fn: Callable[[float], float], lo: float, hi: float,
                       grid_size: int = 81) -> List[float]:
    if hi <= lo:
        return []
    xs = np.linspace(lo, hi, grid_size)
    vals = []
    for x in xs:
        try:
            y = float(fn(float(x)))
        except Exception:
            y = float("nan")
        vals.append(y)
    roots: List[float] = []
    for i in range(len(xs) - 1):
        x0, x1 = float(xs[i]), float(xs[i + 1])
        y0, y1 = vals[i], vals[i + 1]
        if math.isfinite(y0) and abs(y0) <= 1e-8:
            roots.append(x0)
        if not (math.isfinite(y0) and math.isfinite(y1)):
            continue
        if y0 * y1 < 0:
            try:
                roots.append(float(brentq(fn, x0, x1, maxiter=100)))
            except ValueError:
                pass
    if math.isfinite(vals[-1]) and abs(vals[-1]) <= 1e-8:
        roots.append(float(xs[-1]))
    return sorted({round(x, 10) for x in roots})


def _region_is_valid(region: str, p, lam_on: float, lam_off: float,
                     u_on: float, u_off: float, tol: float = 1e-7) -> bool:
    L = p.Lambda
    if lam_on < -tol or lam_off < -tol or lam_on + lam_off > L + tol:
        return False
    if region == "B":
        return lam_on <= tol and lam_off <= tol and u_on <= tol and u_off <= tol
    if region == "V":
        return abs(lam_on - L) <= tol and lam_off <= tol and u_on >= -tol and u_on >= u_off - tol
    if region == "F":
        return lam_on <= tol and abs(lam_off - L) <= tol and u_off >= -tol and u_off >= u_on - tol
    if region == "BV":
        return tol < lam_on < L - tol and lam_off <= tol and abs(u_on) <= tol and u_off <= tol
    if region == "BF":
        return lam_on <= tol and tol < lam_off < L - tol and abs(u_off) <= tol and u_on <= tol
    if region == "VF":
        return (lam_on > tol and lam_off > tol and abs(lam_on + lam_off - L) <= tol
                and abs(u_on - u_off) <= tol and u_on >= -tol)
    if region == "BVF":
        return (lam_on > tol and lam_off > tol and lam_on + lam_off < L - tol
                and abs(u_on) <= tol and abs(u_off) <= tol)
    return False


def solve_policy_equilibrium(p, mu_on: float, p_on: float, p_off: float) -> Optional[Dict[str, Any]]:
    """Return a Wardrop equilibrium for a fixed feasible pricing-capacity policy."""
    mu_off = p.mu_bar - mu_on
    if not (0.0 <= p_on <= p.pbar_on + 1e-9):
        return None
    if isinstance(p, Ch5Params) and not (0.0 <= p_off <= p.pbar_off + 1e-9):
        return None
    if mu_on <= p.eps or mu_off <= p.eps:
        return None

    candidates: List[Dict[str, Any]] = []

    def finalize() -> Optional[Dict[str, Any]]:
        if not candidates:
            return None
        unique: Dict[Tuple[str, float, float], Dict[str, Any]] = {}
        for cand in candidates:
            key = (cand["region"], round(cand["lambda_on"], 5), round(cand["lambda_off"], 5))
            unique[key] = cand
        viable = list(unique.values())
        priority = {"BVF": 0, "VF": 1, "BV": 2, "BF": 3, "V": 4, "F": 5, "B": 6}
        viable.sort(key=lambda c: priority[c["region"]])
        equilibrium = viable[0]
        equilibrium["candidate_count"] = len(viable)
        return equilibrium

    def add(region: str, lam_on: float, lam_off: float) -> None:
        waits = _policy_waits(p, mu_on, mu_off, lam_on, lam_off)
        if waits is None:
            return
        A, B = waits
        u_on, u_off = _policy_utilities(p, p_on, p_off, A, B)
        if _region_is_valid(region, p, lam_on, lam_off, u_on, u_off):
            candidates.append(dict(region=region, lambda_on=lam_on, lambda_off=lam_off,
                                   lambda_balk=max(0.0, p.Lambda - lam_on - lam_off),
                                   A=A, B=B, U_on=u_on, U_off=u_off))

    add("B", 0.0, 0.0)
    add("V", p.Lambda, 0.0)
    add("F", 0.0, p.Lambda)
    if candidates:
        return finalize()

    flow_eps = max(1e-6, p.Lambda * 1e-8)
    lo, hi = flow_eps, p.Lambda - flow_eps

    def utility_on_single(lam_on: float) -> float:
        waits = _policy_waits(p, mu_on, mu_off, lam_on, 0.0)
        if waits is None:
            return float("nan")
        return _policy_utilities(p, p_on, p_off, *waits)[0]

    def utility_off_single(lam_off: float) -> float:
        waits = _policy_waits(p, mu_on, mu_off, 0.0, lam_off)
        if waits is None:
            return float("nan")
        return _policy_utilities(p, p_on, p_off, *waits)[1]

    def utility_difference_full(lam_on: float) -> float:
        waits = _policy_waits(p, mu_on, mu_off, lam_on, p.Lambda - lam_on)
        if waits is None:
            return float("nan")
        u_on, u_off = _policy_utilities(p, p_on, p_off, *waits)
        return u_on - u_off

    for x in _find_scalar_roots(utility_on_single, lo, hi):
        add("BV", x, 0.0)
    for x in _find_scalar_roots(utility_off_single, lo, hi):
        add("BF", 0.0, x)
    for x in _find_scalar_roots(utility_difference_full, lo, hi):
        add("VF", x, p.Lambda - x)
    if candidates:
        return finalize()

    def two_active_residual(z: np.ndarray) -> np.ndarray:
        lam_on, lam_off = float(z[0]), float(z[1])
        waits = _policy_waits(p, mu_on, mu_off, lam_on, lam_off)
        if waits is None:
            return np.array([1e4, 1e4])
        return np.array(_policy_utilities(p, p_on, p_off, *waits))

    seeds = [
        (0.15, 0.15), (0.25, 0.25), (0.45, 0.15),
        (0.15, 0.45), (0.60, 0.20), (0.20, 0.60),
    ]
    for frac_on, frac_off in seeds:
        x0 = np.array([frac_on * p.Lambda, frac_off * p.Lambda])
        try:
            fit = least_squares(two_active_residual, x0,
                                bounds=([flow_eps, flow_eps], [hi, hi]),
                                xtol=1e-11, ftol=1e-11, gtol=1e-11,
                                max_nfev=300)
        except Exception:
            continue
        if float(np.linalg.norm(fit.fun)) <= 2e-5:
            add("BVF", float(fit.x[0]), float(fit.x[1]))

    return finalize()


def _result_for_policy(p, mu_on: float, p_on: float, p_off: float):
    equilibrium = solve_policy_equilibrium(p, mu_on, p_on, p_off)
    if equilibrium is None:
        return None
    if isinstance(p, Ch4Params):
        rev = revenue_ch4(p, p_on, equilibrium["lambda_on"], equilibrium["lambda_off"])
        return Ch4Result(equilibrium["region"], mu_on, p.mu_bar - mu_on, p_on, p_off,
                         equilibrium["lambda_on"], equilibrium["lambda_off"],
                         equilibrium["lambda_balk"], rev, equilibrium)
    rev = revenue_ch5(p, p_on, p_off, equilibrium["lambda_on"], equilibrium["lambda_off"])
    return Ch5Result(equilibrium["region"], mu_on, p.mu_bar - mu_on, p_on, p_off,
                     equilibrium["lambda_on"], equilibrium["lambda_off"],
                     equilibrium["lambda_balk"], rev, equilibrium)


def _total_wait_exposure(p, result) -> float:
    waits = _policy_waits(p, result.mu_on, result.mu_off,
                          result.lambda_on, result.lambda_off)
    if waits is None:
        return float("inf")
    A, B = waits
    w_on = A / p.Cw_on
    w_off = B / p.Cw_off
    load_on = result.lambda_on + p.delta2 * (result.lambda_off + p.lambda_E)
    load_off = result.lambda_off + p.lambda_E + p.delta1 * result.lambda_on
    return load_on * w_on + load_off * w_off


def _choose_display_price_same_outcome(p, result):
    """Choose a deterministic tariff pair within a revenue-equivalent region face."""
    if isinstance(p, Ch4Params):
        return result
    waits = _policy_waits(p, result.mu_on, result.mu_off,
                          result.lambda_on, result.lambda_off)
    if waits is None:
        return result
    A, B = waits
    c_on = p.V + p.psi - A - p.delta1 * (p.T + B) + p.delta1 * p.s
    c_off = p.V + p.s - p.T - B - p.delta2 * A
    Aub: List[List[float]] = []
    bub: List[float] = []
    Aeq: List[List[float]] = []
    beq: List[float] = []

    def u_on_nonpositive():
        Aub.append([-1.0, -p.delta1]); bub.append(-c_on)

    def u_off_nonpositive():
        Aub.append([-p.delta2, -1.0]); bub.append(-c_off)

    def u_on_nonnegative():
        Aub.append([1.0, p.delta1]); bub.append(c_on)

    def u_off_nonnegative():
        Aub.append([p.delta2, 1.0]); bub.append(c_off)

    def on_at_least_off():
        Aub.append([1.0 - p.delta2, p.delta1 - 1.0]); bub.append(c_on - c_off)

    def off_at_least_on():
        Aub.append([p.delta2 - 1.0, 1.0 - p.delta1]); bub.append(c_off - c_on)

    if result.region == "B":
        u_on_nonpositive(); u_off_nonpositive()
    elif result.region == "V":
        u_on_nonnegative(); on_at_least_off()
    elif result.region == "F":
        u_off_nonnegative(); off_at_least_on()
    elif result.region == "BV":
        Aeq.append([1.0, p.delta1]); beq.append(c_on)
        u_off_nonpositive()
    elif result.region == "BF":
        Aeq.append([p.delta2, 1.0]); beq.append(c_off)
        u_on_nonpositive()
    elif result.region == "VF":
        Aeq.append([1.0 - p.delta2, p.delta1 - 1.0]); beq.append(c_on - c_off)
        u_on_nonnegative()
    elif result.region == "BVF":
        Aeq.extend([[1.0, p.delta1], [p.delta2, 1.0]])
        beq.extend([c_on, c_off])

    load_on = result.lambda_on + p.delta2 * (result.lambda_off + p.lambda_E)
    load_off = result.lambda_off + p.lambda_E + p.delta1 * result.lambda_on
    policy_revenue = load_on * result.p_on + load_off * result.p_off
    display_revenue_tol = max(1e-3, abs(policy_revenue) * 1e-8)
    Aub.append([-load_on, -load_off])
    bub.append(-(policy_revenue - display_revenue_tol))
    fit = linprog(
        c=[0.0, -1.0], A_ub=np.asarray(Aub), b_ub=np.asarray(bub),
        A_eq=(np.asarray(Aeq) if Aeq else None),
        b_eq=(np.asarray(beq) if beq else None),
        bounds=[(0.0, p.pbar_on), (0.0, p.pbar_off)], method="highs",
    )
    if not fit.success:
        result.details["display_price_lp_status"] = str(fit.message)
        return result
    p_on, p_off = float(fit.x[0]), float(fit.x[1])
    revenue = revenue_ch5(p, p_on, p_off, result.lambda_on, result.lambda_off)
    details = dict(result.details)
    details["display_price_lp_status"] = str(fit.message)
    details["display_tie_break"] = "maximum offline tariff on the revenue-equivalent face"
    return Ch5Result(result.region, result.mu_on, result.mu_off, p_on, p_off,
                     result.lambda_on, result.lambda_off, result.lambda_balk,
                     revenue, details)


def _solve_policy_space(p, *, fixed_offline: bool, x0: Optional[List[float]] = None,
                        seed: Optional[int] = None, maxiter: int = 100):
    mu_lo = p.delta2 * p.lambda_E + max(p.eps, 1e-5)
    mu_hi = p.mu_bar - p.lambda_E - max(p.eps, 1e-5)
    if mu_hi <= mu_lo:
        raise RuntimeError("Insufficient effective capacity for the compulsory offline-first flow.")
    if fixed_offline:
        bounds = [(mu_lo, mu_hi), (0.0, p.pbar_on)]
    else:
        bounds = [(mu_lo, mu_hi), (0.0, p.pbar_on), (0.0, p.pbar_off)]

    def evaluate(z):
        mu_on = float(z[0])
        p_on = float(z[1])
        p_off = p.p_off if fixed_offline else float(z[2])
        return _result_for_policy(p, mu_on, p_on, p_off)

    def objective(z):
        result = evaluate(z)
        return 1e12 if result is None else -result.revenue

    solver_seed = (20260524 + (4 if fixed_offline else 5)) if seed is None else seed
    options = dict(bounds=bounds, seed=solver_seed,
                   maxiter=maxiter, popsize=14, tol=1e-8, atol=1e-7,
                   polish=True, updating="immediate", workers=1)
    if x0 is not None:
        options["x0"] = np.asarray(x0, dtype=float)
    fit = differential_evolution(objective, **options)
    solution = evaluate(fit.x)
    if solution is None:
        raise RuntimeError("Policy-space optimizer did not locate a feasible equilibrium.")
    primary_revenue = float(solution.revenue)
    revenue_tol = max(1e-4, abs(primary_revenue) * 1e-9)

    def secondary_objective(z):
        result = evaluate(z)
        if result is None:
            return 1e12
        shortfall = max(0.0, primary_revenue - revenue_tol - result.revenue)
        price_tiebreak = 0.0 if fixed_offline else -1e-9 * result.p_off
        return shortfall * 1e8 + _total_wait_exposure(p, result) + price_tiebreak

    tie_options = dict(options)
    tie_options.update(seed=solver_seed + 1000, maxiter=max(60, maxiter // 2),
                       x0=np.asarray(fit.x, dtype=float))
    tie_fit = differential_evolution(secondary_objective, **tie_options)
    tie_solution = evaluate(tie_fit.x)
    if tie_solution is not None and tie_solution.revenue >= primary_revenue - revenue_tol:
        solution = tie_solution
    solution = _choose_display_price_same_outcome(p, solution)
    solution.details["solver"] = "policy_space_differential_evolution"
    solution.details["optimizer_success"] = bool(fit.success)
    solution.details["optimizer_message"] = str(fit.message)
    solution.details["primary_revenue_optimum"] = primary_revenue
    solution.details["tie_break"] = (
        "minimum total expected waiting exposure among revenue-equivalent policies; "
        "maximum displayed-channel tariff for residual ties"
    )
    return solution


def ch4_solve_global(p: Ch4Params) -> Ch4Result:
    x0 = None
    try:
        regional = _regional_ch4_solve_global(p)
        x0 = [regional.mu_on, regional.p_on]
    except Exception:
        pass
    return _solve_policy_space(p, fixed_offline=True, x0=x0)


def ch5_solve_global(p: Ch5Params) -> Ch5Result:
    x0 = None
    try:
        regional = _regional_ch5_solve_global(p)
        x0 = [regional.mu_on, regional.p_on, regional.p_off]
    except Exception:
        pass
    result = _solve_policy_space(p, fixed_offline=False, x0=x0)
    return result


# =============================================================================
# Baselines & unified grids/wrappers
# =============================================================================

MU_BAR_DEFAULT = 972.0  # 54*18 per day
BASE_COMMON = dict(lambda_E=30.0, V=50.0, T=20.0, s=30.0, r_balk=10.0)

# 独立等待成本基线
CWOFF_BASE = 21.0 * 8
CWON_BASE = 14.0 * 8

# ======== ★ 唯一入口：等待成本“网格”与“固定值”集中配置 ========
# 只保留 “Cw_on 固定，Cw_off 扫描”
CW_OFF_GRID = [15 * 8, 18 * 8, 21 * 8, 24 * 8, 27 * 8, 30 * 8]
CW_ON_FIXED_FOR_OFF_SWEEP = 14 * 8
# ================================================================

# Chapter 4: offline price fixed at 40 + p_on cap
BASE_CH4 = dict(
    delta1=0.35, delta2=0.1, p_off=40.0, mu_bar=MU_BAR_DEFAULT, psi=0.0,
    Cw_off=CWOFF_BASE, Cw_on=CWON_BASE, alpha=0.0,
    pbar_on=60.0,
    outer_grid=301, refine_topk=5, **BASE_COMMON
)

# Chapter 5: price caps p_on<=60, p_off<=80
BASE_CH5 = dict(
    delta1=0.35, delta2=0.1, mu_bar=MU_BAR_DEFAULT, psi=0.0,
    pbar_on=60.0, pbar_off=80.0,
    Cw_off=CWOFF_BASE, Cw_on=CWON_BASE, alpha=0.0,
    outer_grid=301, refine_topk=5, **BASE_COMMON
)


_SOLUTION_CACHE: Dict[Tuple[Any, ...], Any] = {}


def run_ch4(Lambda: float, **kw) -> Ch4Result:
    params = {**BASE_CH4, **kw}
    params.update({"Lambda": Lambda})
    key = ("Ch4", tuple(sorted(params.items())))
    if key in _SOLUTION_CACHE:
        return _SOLUTION_CACHE[key]
    p = Ch4Params(**params)
    result = ch4_solve_global(p)
    _SOLUTION_CACHE[key] = result
    return result


def run_ch5(Lambda: float, **kw) -> Ch5Result:
    params = {**BASE_CH5, **kw}
    params.update({"Lambda": Lambda})
    key = ("Ch5", tuple(sorted(params.items())))
    if key in _SOLUTION_CACHE:
        return _SOLUTION_CACHE[key]
    p = Ch5Params(**params)
    result = ch5_solve_global(p)
    _SOLUTION_CACHE[key] = result
    return result


# Sensitivity grids
S_GRID = [15, 20, 25, 30, 35]
DELTA_GRID = [(0.20, 0.05), (0.35, 0.1), (0.50, 0.15)]
LAM_E_GRID = [0, 20, 30, 40]
PSI_ON_GRID = [0, 5, 10, 15, 20, 25]
T_GRID = [15, 20, 25, 30, 35]

# 所有要在“Lambda 轴上作图”的网格
LAMBDA_GRID = sorted([878, 423, 572, 760, 532, 650, 710])

# 要执行“全套灵敏度分析”的多个基线 Lambda
LAMBDA_BASES = [572.0, 878.0]

# 新增：按到达率给定，扫 mu_bar 的网格
MU_BAR_SWEEP_MAP = {
    572.0: [632.0, 742.0, 852.0, 972.0],
    878.0: [972.0, 1082.0, 1192.0, 1302.0],
}


# =============================================================================
# Safe wrapper
# =============================================================================

def safe_run_ch4(**kwargs) -> Optional[Ch4Result]:
    try:
        return run_ch4(**kwargs)
    except Exception:
        return None


def safe_run_ch5(**kwargs) -> Optional[Ch5Result]:
    try:
        return run_ch5(**kwargs)
    except Exception:
        return None


# =============================================================================
# Plotting & CSV writers (English-only labels)  —— 加点标注
# =============================================================================

# =============================================================================
# Revenue/accessibility experiment and stochastic revenue validation support
# =============================================================================

def outcome_metrics(model: str, sol, params) -> Dict[str, Any]:
    waits = _policy_waits(params, sol.mu_on, sol.mu_off, sol.lambda_on, sol.lambda_off)
    if waits is None:
        raise ValueError("Cannot calculate outcome metrics for an unstable policy.")
    A, B = waits
    wait_on = A / params.Cw_on
    wait_off = B / params.Cw_off
    served = sol.lambda_on + sol.lambda_off
    return dict(
        model=model,
        revenue=float(sol.revenue),
        served_eligible=float(served),
        eligible_access_rate=float(served / params.Lambda),
        eligible_balking_rate=float(sol.lambda_balk / params.Lambda),
        wait_online=float(wait_on),
        wait_offline=float(wait_off),
        total_wait_exposure=float(_total_wait_exposure(params, sol)),
        offline_required_wait_exposure=float(wait_off + params.delta2 * wait_on),
    )


def build_revenue_access_comparison(
        grid: Optional[List[float]] = None,
        out_csv_path: str = os.path.join("csv", "revenue_access_comparison.csv")) -> pd.DataFrame:
    """Write policy outcomes and assert that the dual-cap model dominates its subset."""
    grid = LAMBDA_GRID if grid is None else grid
    rows: List[Dict[str, Any]] = []
    for L in grid:
        p4 = Ch4Params(Lambda=L, **BASE_CH4)
        p5 = Ch5Params(Lambda=L, **BASE_CH5)
        r4 = run_ch4(Lambda=L)
        r5 = run_ch5(Lambda=L)
        if p4.p_off <= p5.pbar_off + 1e-9 and r5.revenue + 1e-3 < r4.revenue:
            raise AssertionError(
                f"Dominance check failed at Lambda={L}: dual-tariff revenue "
                f"{r5.revenue:.6f} < fixed-offline revenue {r4.revenue:.6f}."
            )
        gain = r5.revenue - r4.revenue
        for model, sol, params in (("Ch4", r4, p4), ("Ch5", r5, p5)):
            row = outcome_metrics(model, sol, params)
            row.update(
                Lambda=float(L),
                region=sol.region,
                mu_on=float(sol.mu_on),
                mu_off=float(sol.mu_off),
                p_on=float(sol.p_on),
                p_off=float(sol.p_off),
                revenue_gain_dual_vs_fixed=float(gain),
                dominance_pass=bool(gain >= -1e-3),
            )
            rows.append(row)
    df = pd.DataFrame(rows)
    ensure_dir(os.path.dirname(out_csv_path) or ".")
    df.to_csv(out_csv_path, index=False)
    return df


def audit_optimizer_restarts(
        grid: Optional[List[float]] = None,
        seeds: Optional[List[int]] = None,
        out_csv_path: str = os.path.join("csv", "optimizer_restart_audit.csv")) -> pd.DataFrame:
    """Check that independent global-search restarts recover the reported revenues."""
    grid = [572.0, 878.0] if grid is None else grid
    seeds = [20260524, 20260531, 20260607] if seeds is None else seeds
    rows: List[Dict[str, Any]] = []
    for L in grid:
        for model in ("Ch4", "Ch5"):
            p = Ch4Params(Lambda=L, **BASE_CH4) if model == "Ch4" else Ch5Params(Lambda=L, **BASE_CH5)
            results = [
                _solve_policy_space(p, fixed_offline=(model == "Ch4"), seed=seed)
                for seed in seeds
            ]
            revenues = [float(r.revenue) for r in results]
            mu_values = [float(r.mu_on) for r in results]
            p_on_values = [float(r.p_on) for r in results]
            p_off_values = [float(r.p_off) for r in results]
            revenue_range = max(revenues) - min(revenues)
            mu_range = max(mu_values) - min(mu_values)
            p_on_range = max(p_on_values) - min(p_on_values)
            p_off_range = max(p_off_values) - min(p_off_values)
            for seed, sol in zip(seeds, results):
                rows.append(dict(
                    Lambda=L, model=model, seed=seed, region=sol.region,
                    revenue=float(sol.revenue), p_on=float(sol.p_on), p_off=float(sol.p_off),
                    mu_on=float(sol.mu_on), lambda_on=float(sol.lambda_on),
                    lambda_off=float(sol.lambda_off),
                    revenue_range_across_restarts=revenue_range,
                    mu_on_range_across_restarts=mu_range,
                    p_on_range_across_restarts=p_on_range,
                    p_off_range_across_restarts=p_off_range,
                    restart_pass=(revenue_range <= 1e-2),
                    display_policy_pass=(mu_range <= 0.25 and p_on_range <= 1e-2 and p_off_range <= 1e-2),
                ))
    df = pd.DataFrame(rows)
    ensure_dir(os.path.dirname(out_csv_path) or ".")
    df.to_csv(out_csv_path, index=False)
    return df


@dataclass(frozen=True)
class SimConfig:
    Lambda: float
    lambda_E: float
    delta1: float
    delta2: float
    V: float
    T: float
    s: float
    r_balk: float
    mu_on: float
    mu_off: float
    p_on: float
    p_off: float
    chapter: int
    Cw_on: float
    Cw_off: float
    psi: float = 0.0
    alpha: float = 0.0


def _params_for_sim(cfg: SimConfig):
    common = dict(
        Lambda=cfg.Lambda, lambda_E=cfg.lambda_E, delta1=cfg.delta1,
        delta2=cfg.delta2, V=cfg.V, T=cfg.T, s=cfg.s,
        mu_bar=cfg.mu_on + cfg.mu_off, Cw_on=cfg.Cw_on, Cw_off=cfg.Cw_off,
        psi=cfg.psi, r_balk=cfg.r_balk, alpha=cfg.alpha,
        pbar_on=max(BASE_CH4["pbar_on"], cfg.p_on),
    )
    if cfg.chapter == 4:
        return Ch4Params(**common, p_off=cfg.p_off)
    return Ch5Params(**common, pbar_off=max(BASE_CH5["pbar_off"], cfg.p_off))


def _simulate_revenue_rep(cfg: SimConfig, horizon: float, warmup: float, seed: int) -> Dict[str, float]:
    """Monte Carlo episode-volume evaluation conditional on Wardrop routing."""
    p = _params_for_sim(cfg)
    equilibrium = solve_policy_equilibrium(p, cfg.mu_on, cfg.p_on, cfg.p_off)
    if equilibrium is None:
        return {"revenue_d": -1e12}
    duration = max(1.0, float(horizon) - float(warmup))
    rng = np.random.default_rng(seed)
    n_on = int(rng.poisson(equilibrium["lambda_on"] * duration))
    n_off = int(rng.poisson(equilibrium["lambda_off"] * duration))
    n_balk = int(rng.poisson(equilibrium["lambda_balk"] * duration))
    n_required = int(rng.poisson(cfg.lambda_E * duration))
    ref_to_off = int(rng.binomial(n_on, cfg.delta1))
    ref_to_on = int(rng.binomial(n_off + n_required, cfg.delta2))
    online_encounters = n_on + ref_to_on
    offline_encounters = n_off + n_required + ref_to_off
    revenue_d = (
        online_encounters * cfg.p_on + offline_encounters * cfg.p_off
        - n_balk * cfg.r_balk + cfg.alpha * n_on
    ) / duration
    return {
        "revenue_d": float(revenue_d),
        "eligible_served_d": float((n_on + n_off) / duration),
        "eligible_balked_d": float(n_balk / duration),
    }


def batch_simulate_cfgs_parallel(
        cfgs: Dict[str, SimConfig], *, reps: int, horizon: float, warmup: float,
        base_seed: int, max_workers: Optional[int] = None,
        shared_seeds: bool = True) -> Dict[str, Dict[str, Any]]:
    """Reproducible Monte Carlo validation interface used by the gap runner."""
    del max_workers
    outputs: Dict[str, Dict[str, Any]] = {}
    for job_index, (job_id, cfg) in enumerate(cfgs.items()):
        rows = []
        for rep in range(reps):
            seed = base_seed + rep if shared_seeds else base_seed + job_index * reps + rep
            rows.append(_simulate_revenue_rep(cfg, horizon, warmup, seed))
        outputs[job_id] = {
            "rows": rows,
            "mean_stats": {
                "revenue_d": float(statistics.fmean(row["revenue_d"] for row in rows))
            },
        }
    return outputs


def _fmt_price(v: float) -> str:
    return "?" if (v is None or (isinstance(v, float) and not math.isfinite(v))) else f"{int(round(v))}"


def plot_and_save_capacity(df: pd.DataFrame, xcol: str, title: str, png_path: str, xlabel: str):
    fig = plt.figure()
    for m in ["Ch4", "Ch5"]:
        sub = df[(df["model"] == m) & df["mu_frac_on"].notna()].sort_values(xcol)
        if len(sub) > 0:
            sub = sub.drop_duplicates(subset=xcol, keep='last')
            plt.plot(sub[xcol], sub["mu_frac_on"], marker="o", label=m)
            # —— 为每个点添加 (region, p_off, p_on) 标签
            for _, row in sub.iterrows():
                x = row[xcol]; y = row["mu_frac_on"]
                region = row.get("region")
                p_on = row.get("p_on")
                p_off = row.get("p_off")
                if pd.notna(y) and (region is not None):
                    tag = f"({region},{_fmt_price(p_off)},{_fmt_price(p_on)})"
                    plt.annotate(tag, (x, y), textcoords="offset points", xytext=(5, 6), fontsize=8)
    plt.xlabel(xlabel)
    plt.ylabel("mu_v / mu_total")
    plt.title(title)
    plt.grid(True)
    _disable_axis_offsets(plt.gca())
    plt.legend()
    save_and_show(fig, png_path)


def plot_and_save_ch4_price(df: pd.DataFrame, xcol: str, title: str, png_path: str, xlabel: str):
    fig = plt.figure()
    price_values = []
    sub = df[(df["model"] == "Ch4") & df["p_on"].notna()].sort_values(xcol)
    if len(sub) > 0:
        sub = sub.drop_duplicates(subset=xcol, keep='last')
        price_values = list(sub["p_on"])
        plt.plot(
            sub[xcol], sub["p_on"],
            color=CH4_PRICE_COLOR,
            marker=CH4_PRICE_MARKER,
            linewidth=PRICE_LINEWIDTH,
            markersize=PRICE_MARKERSIZE,
            label="p_on (Ch4)",
        )
    plt.xlabel(xlabel)
    plt.ylabel("Price")
    plt.title(title)
    plt.grid(True)
    _format_price_axis(plt.gca(), price_values)
    plt.legend()
    save_and_show(fig, png_path)


def plot_and_save_ch5_price(df: pd.DataFrame, xcol: str, title: str, png_path: str, xlabel: str):
    fig = plt.figure()
    price_values = []
    sub = df[(df["model"] == "Ch5") & df["p_on"].notna()].sort_values(xcol)
    if len(sub) > 0:
        sub = sub.drop_duplicates(subset=xcol, keep='last')
        price_values = list(sub["p_on"]) + list(sub["p_off"])
        plt.plot(
            sub[xcol], sub["p_on"],
            color=CH5_PON_COLOR,
            marker=CH5_PRICE_MARKER,
            linewidth=PRICE_LINEWIDTH,
            markersize=PRICE_MARKERSIZE,
            label="p_on (Ch5)",
        )
        plt.plot(
            sub[xcol], sub["p_off"],
            color=CH5_POFF_COLOR,
            marker=CH5_PRICE_MARKER,
            linewidth=PRICE_LINEWIDTH,
            markersize=PRICE_MARKERSIZE,
            label="p_off (Ch5)",
        )
    plt.xlabel(xlabel)
    plt.ylabel("Price")
    plt.title(title)
    plt.grid(True)
    _format_price_axis(plt.gca(), price_values)
    plt.legend()
    save_and_show(fig, png_path)


# ----- Records helpers WITH region & prices for capacity plot annotations -----

def to_records_for_mu_share(label, model, result, mu_bar=MU_BAR_DEFAULT):
    mu_on = None if (result is None) else result.mu_on
    region = None if (result is None) else result.region
    p_on = None if (result is None) else result.p_on
    p_off = None if (result is None) else result.p_off
    return dict(
        label=label,
        model=model,
        mu_frac_on=(np.nan if mu_on is None else float(mu_on / mu_bar)),
        region=region,
        p_on=(np.nan if p_on is None else float(p_on)),
        p_off=(np.nan if p_off is None else float(p_off)),
    )


def to_records_for_price(label, model, result):
    p_on = None if (result is None) else result.p_on
    p_off = None if (result is None) else result.p_off
    region = None if (result is None) else result.region
    return dict(
        label=label,
        model=model,
        p_on=(np.nan if p_on is None else float(p_on)),
        p_off=(np.nan if p_off is None else float(p_off)),
        region=region
    )


# ----- Sensitivity: Lambda (总到达率) -----

def sweep_Lambda(grid: List[float], out_csv_dir: str, out_fig_dir: str):
    rows_cap, rows_ch4, rows_ch5 = [], [], []
    for L in grid:
        r4 = safe_run_ch4(Lambda=L)
        r5 = safe_run_ch5(Lambda=L)
        rows_cap += [
            to_records_for_mu_share(L, "Ch4", r4),
            to_records_for_mu_share(L, "Ch5", r5),
        ]
        rows_ch4 += [to_records_for_price(L, "Ch4", r4)]
        rows_ch5 += [to_records_for_price(L, "Ch5", r5)]
    cap_df = pd.DataFrame(rows_cap).rename(columns={"label": "Lambda"})
    ch4_df = pd.DataFrame(rows_ch4).rename(columns={"label": "Lambda"})
    ch5_df = pd.DataFrame(rows_ch5).rename(columns={"label": "Lambda"})
    ensure_dir(out_csv_dir); ensure_dir(out_fig_dir)
    cap_df.to_csv(os.path.join(out_csv_dir, "lambda_capacity_share.csv"), index=False)
    ch4_df.to_csv(os.path.join(out_csv_dir, "lambda_ch4_price.csv"), index=False)
    ch5_df.to_csv(os.path.join(out_csv_dir, "lambda_ch5_price.csv"), index=False)

    plot_and_save_capacity(cap_df.rename(columns={"Lambda": "x"}), "x",
                           "Online share vs Lambda",
                           os.path.join(out_fig_dir, "lambda_capacity_share.png"),
                           "Lambda (total arrival rate)")
    plot_and_save_ch4_price(ch4_df.rename(columns={"Lambda": "x"}), "x",
                            "Fixed offline-tariff regime price vs Lambda",
                            os.path.join(out_fig_dir, "lambda_ch4_price.png"),
                            "Lambda (total arrival rate)")
    plot_and_save_ch5_price(ch5_df.rename(columns={"Lambda": "x"}), "x",
                            "Dual-tariff regime prices vs Lambda",
                            os.path.join(out_fig_dir, "lambda_ch5_price.png"),
                            "Lambda (total arrival rate)")


# ----- Sensitivity: Cw_off sweep（保持 Cw_on 固定） -----

def sweep_Cw_off_given_on(Lambda: float, Cw_on_fixed: float = None,
                          grid_off: List[float] = None,
                          out_csv_dir: str = "csv", out_fig_dir: str = "figs"):
    if Cw_on_fixed is None:
        Cw_on_fixed = CW_ON_FIXED_FOR_OFF_SWEEP
    if grid_off is None:
        grid_off = CW_OFF_GRID
    rows_cap, rows_ch4, rows_ch5 = [], [], []
    for Cw_off in grid_off:
        r4 = safe_run_ch4(Lambda=Lambda, Cw_off=Cw_off, Cw_on=Cw_on_fixed)
        r5 = safe_run_ch5(Lambda=Lambda, Cw_off=Cw_off, Cw_on=Cw_on_fixed)
        rows_cap += [
            to_records_for_mu_share(Cw_off, "Ch4", r4),
            to_records_for_mu_share(Cw_off, "Ch5", r5),
        ]
        rows_ch4 += [to_records_for_price(Cw_off, "Ch4", r4)]
        rows_ch5 += [to_records_for_price(Cw_off, "Ch5", r5)]
    cap_df = pd.DataFrame(rows_cap).rename(columns={"label": "Cw_off"})
    ch4_df = pd.DataFrame(rows_ch4).rename(columns={"label": "Cw_off"})
    ch5_df = pd.DataFrame(rows_ch5).rename(columns={"label": "Cw_off"})
    ensure_dir(out_csv_dir); ensure_dir(out_fig_dir)
    cap_df.to_csv(os.path.join(out_csv_dir, f"cw_off_given_on_{int(Cw_on_fixed)}_capacity_share.csv"), index=False)
    ch4_df.to_csv(os.path.join(out_csv_dir, f"cw_off_given_on_{int(Cw_on_fixed)}_ch4_price.csv"), index=False)
    ch5_df.to_csv(os.path.join(out_csv_dir, f"cw_off_given_on_{int(Cw_on_fixed)}_ch5_price.csv"), index=False)

    plot_and_save_capacity(cap_df.rename(columns={"Cw_off": "x"}), "x",
                           f"Online share vs Cw_off (Lambda={Lambda}, Cw_on={Cw_on_fixed})",
                           os.path.join(out_fig_dir, f"cw_off_given_on_{int(Cw_on_fixed)}_capacity_share.png"),
                           "Cw_off (offline waiting cost per hour)")
    plot_and_save_ch4_price(ch4_df.rename(columns={"Cw_off": "x"}), "x",
                            f"Fixed offline-tariff regime price vs Cw_off (Lambda={Lambda}, Cw_on={Cw_on_fixed})",
                            os.path.join(out_fig_dir, f"cw_off_given_on_{int(Cw_on_fixed)}_ch4_price.png"),
                            "Cw_off (offline waiting cost per hour)")
    plot_and_save_ch5_price(ch5_df.rename(columns={"Cw_off": "x"}), "x",
                            f"Dual-tariff regime prices vs Cw_off (Lambda={Lambda}, Cw_on={Cw_on_fixed})",
                            os.path.join(out_fig_dir, f"cw_off_given_on_{int(Cw_on_fixed)}_ch5_price.png"),
                            "Cw_off (offline waiting cost per hour)")


# ----- Sensitivity: s -----

def sweep_s(Lambda: float, grid: List[float], out_csv_dir: str, out_fig_dir: str):
    rows_cap, rows_ch4, rows_ch5 = [], [], []
    for s in grid:
        r4 = safe_run_ch4(Lambda=Lambda, s=s)
        r5 = safe_run_ch5(Lambda=Lambda, s=s)
        rows_cap += [
            to_records_for_mu_share(s, "Ch4", r4),
            to_records_for_mu_share(s, "Ch5", r5),
        ]
        rows_ch4 += [to_records_for_price(s, "Ch4", r4)]
        rows_ch5 += [to_records_for_price(s, "Ch5", r5)]
    cap_df = pd.DataFrame(rows_cap).rename(columns={"label": "s"})
    ch4_df = pd.DataFrame(rows_ch4).rename(columns={"label": "s"})
    ch5_df = pd.DataFrame(rows_ch5).rename(columns={"label": "s"})
    ensure_dir(out_csv_dir); ensure_dir(out_fig_dir)
    cap_df.to_csv(os.path.join(out_csv_dir, "s_capacity_share.csv"), index=False)
    ch4_df.to_csv(os.path.join(out_csv_dir, "s_ch4_price.csv"), index=False)
    ch5_df.to_csv(os.path.join(out_csv_dir, "s_ch5_price.csv"), index=False)

    plot_and_save_capacity(cap_df.rename(columns={"s": "x"}), "x",
                           f"Online share vs s (Lambda={Lambda})",
                           os.path.join(out_fig_dir, "s_capacity_share.png"),
                           "s (offline extra utility)")
    plot_and_save_ch4_price(ch4_df.rename(columns={"s": "x"}), "x",
                            f"Fixed offline-tariff regime price vs s (Lambda={Lambda})",
                            os.path.join(out_fig_dir, "s_ch4_price.png"),
                            "s (offline extra utility)")
    plot_and_save_ch5_price(ch5_df.rename(columns={"s": "x"}), "x",
                            f"Dual-tariff regime prices vs s (Lambda={Lambda})",
                            os.path.join(out_fig_dir, "s_ch5_price.png"),
                            "s (offline extra utility)")


# ----- Sensitivity: (delta1, delta2)  —— 能力图也加标签 -----

def sweep_delta_pairs(Lambda: float, pairs: List[Tuple[float, float]], out_csv_dir: str, out_fig_dir: str):
    rows_cap, rows_ch4, rows_ch5 = [], [], []
    labels = [f"({d1:.2f},{d2:.2f})" for d1, d2 in pairs]
    for (d1, d2), lab in zip(pairs, labels):
        r4 = safe_run_ch4(Lambda=Lambda, delta1=d1, delta2=d2)
        r5 = safe_run_ch5(Lambda=Lambda, delta1=d1, delta2=d2)
        rows_cap += [
            dict(pair=lab, model="Ch4",
                 mu_frac_on=(np.nan if r4 is None else r4.mu_on / MU_BAR_DEFAULT),
                 region=(None if r4 is None else r4.region),
                 p_on=(np.nan if r4 is None else r4.p_on),
                 p_off=(np.nan if r4 is None else r4.p_off)),
            dict(pair=lab, model="Ch5",
                 mu_frac_on=(np.nan if r5 is None else r5.mu_on / MU_BAR_DEFAULT),
                 region=(None if r5 is None else r5.region),
                 p_on=(np.nan if r5 is None else r5.p_on),
                 p_off=(np.nan if r5 is None else r5.p_off)),
        ]
        rows_ch4 += [dict(pair=lab, model="Ch4",
                          p_on=(np.nan if r4 is None else r4.p_on),
                          region=(None if r4 is None else r4.region))]
        rows_ch5 += [dict(pair=lab, model="Ch5",
                          p_on=(np.nan if r5 is None else r5.p_on),
                          p_off=(np.nan if r5 is None else r5.p_off),
                          region=(None if r5 is None else r5.region))]
    cap_df = pd.DataFrame(rows_cap)
    ch4_df = pd.DataFrame(rows_ch4)
    ch5_df = pd.DataFrame(rows_ch5)
    ensure_dir(out_csv_dir); ensure_dir(out_fig_dir)
    cap_df.to_csv(os.path.join(out_csv_dir, "delta_capacity_share.csv"), index=False)
    ch4_df.to_csv(os.path.join(out_csv_dir, "delta_ch4_price.csv"), index=False)
    ch5_df.to_csv(os.path.join(out_csv_dir, "delta_ch5_price.csv"), index=False)

    xs = np.arange(len(labels))
    # capacity + 标签
    fig = plt.figure()
    for m in ["Ch4", "Ch5"]:
        sub = cap_df[(cap_df["model"] == m) & cap_df["mu_frac_on"].notna()]
        if len(sub) > 0:
            y = [float(sub[sub["pair"] == lab]["mu_frac_on"].values[0]) if lab in sub["pair"].values else np.nan
                 for lab in labels]
            plt.plot(xs, y, marker="o", label=m)
            for i, lab in enumerate(labels):
                row = sub[sub["pair"] == lab]
                if len(row) == 1 and pd.notna(y[i]):
                    region = row["region"].values[0]
                    p_on = row["p_on"].values[0]
                    p_off = row["p_off"].values[0]
                    tag = f"({region},{_fmt_price(p_off)},{_fmt_price(p_on)})"
                    plt.annotate(tag, (xs[i], y[i]), textcoords="offset points", xytext=(5, 6), fontsize=8)
    plt.xticks(xs, labels)
    plt.xlabel("Referral probabilities (delta1, delta2)")
    plt.ylabel("mu_v / mu_total")
    plt.title(f"Online share vs referrals (Lambda={Lambda})")
    plt.grid(True); _disable_axis_offsets(plt.gca()); plt.legend()
    save_and_show(fig, os.path.join(out_fig_dir, "delta_capacity_share.png"))

    # ch4 price
    fig = plt.figure()
    price_values = []
    sub = ch4_df[ch4_df["p_on"].notna()]
    if len(sub) > 0:
        y = [float(sub[sub["pair"] == lab]["p_on"].values[0]) if lab in sub["pair"].values else np.nan
             for lab in labels]
        price_values = y
        plt.plot(
            xs, y,
            color=CH4_PRICE_COLOR,
            marker=CH4_PRICE_MARKER,
            linewidth=PRICE_LINEWIDTH,
            markersize=PRICE_MARKERSIZE,
            label="p_on (Ch4)",
        )
    plt.xticks(xs, labels)
    plt.xlabel("Referral probabilities (delta1, delta2)")
    plt.ylabel("Price")
    plt.title(f"Fixed offline-tariff regime price vs referrals (Lambda={Lambda})")
    plt.grid(True); _format_price_axis(plt.gca(), price_values); plt.legend()
    save_and_show(fig, os.path.join(out_fig_dir, "delta_ch4_price.png"))

    # ch5 prices
    fig = plt.figure()
    price_values = []
    sub = ch5_df[ch5_df["p_on"].notna()]
    if len(sub) > 0:
        mp_on = {str(row["pair"]): float(row["p_on"]) for _, row in sub.iterrows()}
        mp_off = {str(row["pair"]): float(row["p_off"]) for _, row in sub.iterrows()}
        y_on = [mp_on.get(lab, np.nan) for lab in labels]
        y_off = [mp_off.get(lab, np.nan) for lab in labels]
        price_values = y_on + y_off
        plt.plot(
            xs, y_on,
            color=CH5_PON_COLOR,
            marker=CH5_PRICE_MARKER,
            linewidth=PRICE_LINEWIDTH,
            markersize=PRICE_MARKERSIZE,
            label="p_on (Ch5)",
        )
        plt.plot(
            xs, y_off,
            color=CH5_POFF_COLOR,
            marker=CH5_PRICE_MARKER,
            linewidth=PRICE_LINEWIDTH,
            markersize=PRICE_MARKERSIZE,
            label="p_off (Ch5)",
        )
    plt.xticks(xs, labels)
    plt.xlabel("Referral probabilities (delta1, delta2)")
    plt.ylabel("Price")
    plt.title(f"Dual-tariff regime prices vs referrals (Lambda={Lambda})")
    plt.grid(True); _format_price_axis(plt.gca(), price_values); plt.legend()
    save_and_show(fig, os.path.join(out_fig_dir, "delta_ch5_price.png"))


# ----- Sensitivity: lambda_E -----

def sweep_lambdaE(Lambda: float, grid: List[float], out_csv_dir: str, out_fig_dir: str):
    rows_cap, rows_ch4, rows_ch5 = [], [], []
    for lamE in grid:
        r4 = safe_run_ch4(Lambda=Lambda, lambda_E=lamE)
        r5 = safe_run_ch5(Lambda=Lambda, lambda_E=lamE)
        rows_cap += [
            to_records_for_mu_share(lamE, "Ch4", r4),
            to_records_for_mu_share(lamE, "Ch5", r5),
        ]
        rows_ch4 += [to_records_for_price(lamE, "Ch4", r4)]
        rows_ch5 += [to_records_for_price(lamE, "Ch5", r5)]
    cap_df = pd.DataFrame(rows_cap).rename(columns={"label": "lambda_E"})
    ch4_df = pd.DataFrame(rows_ch4).rename(columns={"label": "lambda_E"})
    ch5_df = pd.DataFrame(rows_ch5).rename(columns={"label": "lambda_E"})
    ensure_dir(out_csv_dir); ensure_dir(out_fig_dir)
    cap_df.to_csv(os.path.join(out_csv_dir, "lambdaE_capacity_share.csv"), index=False)
    ch4_df.to_csv(os.path.join(out_csv_dir, "lambdaE_ch4_price.csv"), index=False)
    ch5_df.to_csv(os.path.join(out_csv_dir, "lambdaE_ch5_price.csv"), index=False)

    plot_and_save_capacity(cap_df.rename(columns={"lambda_E": "x"}), "x",
                           f"Online share vs lambda_E (Lambda={Lambda})",
                           os.path.join(out_fig_dir, "lambdaE_capacity_share.png"),
                           "lambda_E (exogenous emergency load)")
    plot_and_save_ch4_price(ch4_df.rename(columns={"lambda_E": "x"}), "x",
                            f"Fixed offline-tariff regime price vs lambda_E (Lambda={Lambda})",
                            os.path.join(out_fig_dir, "lambdaE_ch4_price.png"),
                            "lambda_E (exogenous emergency load)")
    plot_and_save_ch5_price(ch5_df.rename(columns={"lambda_E": "x"}), "x",
                            f"Dual-tariff regime prices vs lambda_E (Lambda={Lambda})",
                            os.path.join(out_fig_dir, "lambdaE_ch5_price.png"),
                            "lambda_E (exogenous emergency load)")


# ----- Sensitivity: T (transportation / access cost) -----

def sweep_T(Lambda: float, grid: List[float], out_csv_dir: str, out_fig_dir: str):
    rows_cap, rows_ch4, rows_ch5 = [], [], []
    for T_val in grid:
        r4 = safe_run_ch4(Lambda=Lambda, T=T_val)
        r5 = safe_run_ch5(Lambda=Lambda, T=T_val)
        rows_cap += [
            to_records_for_mu_share(T_val, "Ch4", r4),
            to_records_for_mu_share(T_val, "Ch5", r5),
        ]
        rows_ch4 += [to_records_for_price(T_val, "Ch4", r4)]
        rows_ch5 += [to_records_for_price(T_val, "Ch5", r5)]
    cap_df = pd.DataFrame(rows_cap).rename(columns={"label": "T"})
    ch4_df = pd.DataFrame(rows_ch4).rename(columns={"label": "T"})
    ch5_df = pd.DataFrame(rows_ch5).rename(columns={"label": "T"})
    ensure_dir(out_csv_dir); ensure_dir(out_fig_dir)
    cap_df.to_csv(os.path.join(out_csv_dir, "T_capacity_share.csv"), index=False)
    ch4_df.to_csv(os.path.join(out_csv_dir, "T_ch4_price.csv"), index=False)
    ch5_df.to_csv(os.path.join(out_csv_dir, "T_ch5_price.csv"), index=False)

    plot_and_save_capacity(cap_df.rename(columns={"T": "x"}), "x",
                           f"Online share vs T (Lambda={Lambda})",
                           os.path.join(out_fig_dir, "T_capacity_share.png"),
                           "T (transport/access cost)")
    plot_and_save_ch4_price(ch4_df.rename(columns={"T": "x"}), "x",
                            f"Fixed offline-tariff regime price vs T (Lambda={Lambda})",
                            os.path.join(out_fig_dir, "T_ch4_price.png"),
                            "T (transport/access cost)")
    plot_and_save_ch5_price(ch5_df.rename(columns={"T": "x"}), "x",
                            f"Dual-tariff regime prices vs T (Lambda={Lambda})",
                            os.path.join(out_fig_dir, "T_ch5_price.png"),
                            "T (transport/access cost)")

# ----- Sensitivity: psi_on（线上额外效用） -----

def sweep_psi_on(Lambda: float, grid: List[float],
                 out_csv_dir: str, out_fig_dir: str):
    rows_cap, rows_ch4, rows_ch5 = [], [], []
    for psi in grid:
        r4 = safe_run_ch4(Lambda=Lambda, psi=psi)
        r5 = safe_run_ch5(Lambda=Lambda, psi=psi)
        rows_cap += [
            to_records_for_mu_share(psi, "Ch4", r4),
            to_records_for_mu_share(psi, "Ch5", r5),
        ]
        rows_ch4 += [to_records_for_price(psi, "Ch4", r4)]
        rows_ch5 += [to_records_for_price(psi, "Ch5", r5)]

    cap_df = pd.DataFrame(rows_cap).rename(columns={"label": "psi_on"})
    ch4_df = pd.DataFrame(rows_ch4).rename(columns={"label": "psi_on"})
    ch5_df = pd.DataFrame(rows_ch5).rename(columns={"label": "psi_on"})

    ensure_dir(out_csv_dir)
    ensure_dir(out_fig_dir)
    cap_df.to_csv(os.path.join(out_csv_dir, "psi_on_capacity_share.csv"), index=False)
    ch4_df.to_csv(os.path.join(out_csv_dir, "psi_on_ch4_price.csv"), index=False)
    ch5_df.to_csv(os.path.join(out_csv_dir, "psi_on_ch5_price.csv"), index=False)

    plot_and_save_capacity(
        cap_df.rename(columns={"psi_on": "x"}), "x",
        f"Online share vs psi_on (Lambda={Lambda})",
        os.path.join(out_fig_dir, "psi_on_capacity_share.png"),
        "psi_on (online extra utility)"
    )
    plot_and_save_ch4_price(
        ch4_df.rename(columns={"psi_on": "x"}), "x",
        f"Fixed offline-tariff regime price vs psi_on (Lambda={Lambda})",
        os.path.join(out_fig_dir, "psi_on_ch4_price.png"),
        "psi_on (online extra utility)"
    )
    plot_and_save_ch5_price(
        ch5_df.rename(columns={"psi_on": "x"}), "x",
        f"Dual-tariff regime prices vs psi_on (Lambda={Lambda})",
        os.path.join(out_fig_dir, "psi_on_ch5_price.png"),
        "psi_on (online extra utility)"
    )



# ----- NEW: Sensitivity — mu_bar given Lambda -----

def sweep_mu_bar_given_lambda(Lambda: float, mu_grid: List[float],
                              out_csv_dir: str = "csv", out_fig_dir: str = "figs"):
    rows_cap, rows_ch4, rows_ch5 = [], [], []
    for mu_val in mu_grid:
        r4 = safe_run_ch4(Lambda=Lambda, mu_bar=mu_val)
        r5 = safe_run_ch5(Lambda=Lambda, mu_bar=mu_val)
        rows_cap += [
            to_records_for_mu_share(mu_val, "Ch4", r4, mu_bar=mu_val),
            to_records_for_mu_share(mu_val, "Ch5", r5, mu_bar=mu_val),
        ]
        rows_ch4 += [to_records_for_price(mu_val, "Ch4", r4)]
        rows_ch5 += [to_records_for_price(mu_val, "Ch5", r5)]
    cap_df = pd.DataFrame(rows_cap).rename(columns={"label": "mu_bar"})
    ch4_df = pd.DataFrame(rows_ch4).rename(columns={"label": "mu_bar"})
    ch5_df = pd.DataFrame(rows_ch5).rename(columns={"label": "mu_bar"})
    ensure_dir(out_csv_dir); ensure_dir(out_fig_dir)
    cap_df.to_csv(os.path.join(out_csv_dir, f"mu_bar_L{int(Lambda)}_capacity_share.csv"), index=False)
    ch4_df.to_csv(os.path.join(out_csv_dir, f"mu_bar_L{int(Lambda)}_ch4_price.csv"), index=False)
    ch5_df.to_csv(os.path.join(out_csv_dir, f"mu_bar_L{int(Lambda)}_ch5_price.csv"), index=False)

    plot_and_save_capacity(cap_df.rename(columns={"mu_bar": "x"}), "x",
                           f"Online share vs mu_bar (Lambda={Lambda})",
                           os.path.join(out_fig_dir, f"mu_bar_L{int(Lambda)}_capacity_share.png"),
                           "mu_total (total capacity per day)")
    plot_and_save_ch4_price(ch4_df.rename(columns={"mu_bar": "x"}), "x",
                            f"Fixed offline-tariff regime price vs mu_bar (Lambda={Lambda})",
                            os.path.join(out_fig_dir, f"mu_bar_L{int(Lambda)}_ch4_price.png"),
                            "mu_total (total capacity per day)")
    plot_and_save_ch5_price(ch5_df.rename(columns={"mu_bar": "x"}), "x",
                            f"Dual-tariff regime prices vs mu_bar (Lambda={Lambda})",
                            os.path.join(out_fig_dir, f"mu_bar_L{int(Lambda)}_ch5_price.png"),
                            "mu_total (total capacity per day)")


# ========= Master table (single CSV for all sensitivities) =========

def _base_params_for_model(model: str) -> dict:
    if model == "Ch4":
        return dict(
            lambda_E=BASE_CH4.get("lambda_E", BASE_COMMON["lambda_E"]),
            V=BASE_CH4.get("V", BASE_COMMON["V"]),
            T=BASE_CH4.get("T", BASE_COMMON["T"]),
            s=BASE_CH4.get("s", BASE_COMMON["s"]),
            delta1=BASE_CH4.get("delta1", 0.35),
            delta2=BASE_CH4.get("delta2", 0.15),
            psi=BASE_CH4.get("psi", 0.0),
            mu_bar=BASE_CH4.get("mu_bar", MU_BAR_DEFAULT),
            Cw_on=BASE_CH4.get("Cw_on", CWON_BASE),
            Cw_off=BASE_CH4.get("Cw_off", CWOFF_BASE),
            alpha=BASE_CH4.get("alpha", 0.0),
            p_off_param=BASE_CH4.get("p_off", 40.0),
            pbar_on=BASE_CH4.get("pbar_on", 60.0),
            pbar_off=float("nan"),
        )
    else:  # Ch5
        return dict(
            lambda_E=BASE_CH5.get("lambda_E", BASE_COMMON["lambda_E"]),
            V=BASE_CH5.get("V", BASE_COMMON["V"]),
            T=BASE_CH5.get("T", BASE_COMMON["T"]),
            s=BASE_CH5.get("s", BASE_COMMON["s"]),
            delta1=BASE_CH5.get("delta1", 0.35),
            delta2=BASE_CH5.get("delta2", 0.15),
            psi=BASE_CH5.get("psi", 0.0),
            mu_bar=BASE_CH5.get("mu_bar", MU_BAR_DEFAULT),
            Cw_on=BASE_CH5.get("Cw_on", CWON_BASE),
            Cw_off=BASE_CH5.get("Cw_off", CWOFF_BASE),
            alpha=BASE_CH5.get("alpha", 0.0),
            p_off_param=float("nan"),
            pbar_on=BASE_CH5.get("pbar_on", 60.0),
            pbar_off=BASE_CH5.get("pbar_off", 80.0),
        )


def _row_from_solution(model: str, sol, Lambda_val: float, param_overrides: dict) -> dict:
    base = _base_params_for_model(model)
    base.update(param_overrides or {})
    base["Lambda"] = Lambda_val
    out = dict(
        model=model,
        region=(getattr(sol, "region", None) if sol is not None else None),
        p_on=(None if sol is None else float(sol.p_on)),
        p_off=(None if sol is None else float(sol.p_off)),
        mu_on=(None if sol is None else float(sol.mu_on)),
        mu_off=(None if sol is None else float(sol.mu_off)),
        lambda_on=(None if sol is None else float(sol.lambda_on)),
        lambda_off=(None if sol is None else float(sol.lambda_off)),
        lambda_balk=(None if sol is None else float(sol.lambda_balk)),
        revenue=(None if sol is None else float(sol.revenue)),
    )
    return {**out, **base}


def build_master_table_and_save(
        lambda_bases: List[float] = LAMBDA_BASES,
        out_csv_path: str = os.path.join("csv", "master_sensitivity_results.csv"),
        dedup: bool = True
):
    rows = []

    def _safe(model: str, **kw):
        try:
            return run_ch4(**kw) if model == "Ch4" else run_ch5(**kw)
        except Exception:
            return None

    # (1) Lambda sweep
    for L in LAMBDA_GRID:
        sol4 = _safe("Ch4", Lambda=L)
        sol5 = _safe("Ch5", Lambda=L)
        rows.append(_row_from_solution("Ch4", sol4, L, {}))
        rows.append(_row_from_solution("Ch5", sol5, L, {}))

    # (2)-(?) 对每个基线 Lambda 都完整跑一遍
    for Lambda_base in lambda_bases:
        # (2) Cw_off sweep @ Cw_on fixed = CW_ON_FIXED_FOR_OFF_SWEEP
        for Cw_off in CW_OFF_GRID:
            sol4 = _safe("Ch4", Lambda=Lambda_base, Cw_off=Cw_off, Cw_on=CW_ON_FIXED_FOR_OFF_SWEEP)
            sol5 = _safe("Ch5", Lambda=Lambda_base, Cw_off=Cw_off, Cw_on=CW_ON_FIXED_FOR_OFF_SWEEP)
            rows.append(_row_from_solution("Ch4", sol4, Lambda_base, {"Cw_off": Cw_off, "Cw_on": CW_ON_FIXED_FOR_OFF_SWEEP}))
            rows.append(_row_from_solution("Ch5", sol5, Lambda_base, {"Cw_off": Cw_off, "Cw_on": CW_ON_FIXED_FOR_OFF_SWEEP}))

        # (3) s sweep
        for s in S_GRID:
            sol4 = _safe("Ch4", Lambda=Lambda_base, s=s)
            sol5 = _safe("Ch5", Lambda=Lambda_base, s=s)
            rows.append(_row_from_solution("Ch4", sol4, Lambda_base, {"s": s}))
            rows.append(_row_from_solution("Ch5", sol5, Lambda_base, {"s": s}))

        # (4) (delta1, delta2) sweep
        for (d1, d2) in DELTA_GRID:
            sol4 = _safe("Ch4", Lambda=Lambda_base, delta1=d1, delta2=d2)
            sol5 = _safe("Ch5", Lambda=Lambda_base, delta1=d1, delta2=d2)
            rows.append(_row_from_solution("Ch4", sol4, Lambda_base, {"delta1": d1, "delta2": d2}))
            rows.append(_row_from_solution("Ch5", sol5, Lambda_base, {"delta1": d1, "delta2": d2}))

        # (5) lambda_E sweep
        for lamE in LAM_E_GRID:
            sol4 = _safe("Ch4", Lambda=Lambda_base, lambda_E=lamE)
            sol5 = _safe("Ch5", Lambda=Lambda_base, lambda_E=lamE)
            rows.append(_row_from_solution("Ch4", sol4, Lambda_base, {"lambda_E": lamE}))
            rows.append(_row_from_solution("Ch5", sol5, Lambda_base, {"lambda_E": lamE}))

        # (6) T sweep
        for T_val in T_GRID:
            sol4 = _safe("Ch4", Lambda=Lambda_base, T=T_val)
            sol5 = _safe("Ch5", Lambda=Lambda_base, T=T_val)
            rows.append(_row_from_solution("Ch4", sol4, Lambda_base, {"T": T_val}))
            rows.append(_row_from_solution("Ch5", sol5, Lambda_base, {"T": T_val}))

        # (7) psi_on sweep
        for psi_on in PSI_ON_GRID:
            sol4 = _safe("Ch4", Lambda=Lambda_base, psi=psi_on)
            sol5 = _safe("Ch5", Lambda=Lambda_base, psi=psi_on)
            rows.append(_row_from_solution("Ch4", sol4, Lambda_base, {"psi": psi_on}))
            rows.append(_row_from_solution("Ch5", sol5, Lambda_base, {"psi": psi_on}))

        # (8) NEW: mu_bar sweep for this Lambda
        if Lambda_base in MU_BAR_SWEEP_MAP:
            for mu_val in MU_BAR_SWEEP_MAP[Lambda_base]:
                sol4 = _safe("Ch4", Lambda=Lambda_base, mu_bar=mu_val)
                sol5 = _safe("Ch5", Lambda=Lambda_base, mu_bar=mu_val)
                rows.append(_row_from_solution("Ch4", sol4, Lambda_base, {"mu_bar": mu_val}))
                rows.append(_row_from_solution("Ch5", sol5, Lambda_base, {"mu_bar": mu_val}))

    df = pd.DataFrame(rows)

    col_order = [
        "model", "region",
        "Lambda", "lambda_E", "delta1", "delta2", "V", "T", "s", "psi",
        "mu_bar", "Cw_on", "Cw_off", "alpha",
        "p_off_param", "pbar_on", "pbar_off",
        "p_on", "p_off", "mu_on", "mu_off",
        "lambda_on", "lambda_off", "lambda_balk", "revenue",
    ]
    col_order += [c for c in df.columns if c not in col_order]
    df = df[col_order]

    if dedup:
        sig = ["model", "Lambda", "lambda_E", "delta1", "delta2", "V", "T", "s", "psi",
               "mu_bar", "Cw_on", "Cw_off", "alpha", "p_off_param", "pbar_on", "pbar_off"]
        df = df.sort_values(sig).drop_duplicates(subset=sig, keep="first")

    out_dir = os.path.dirname(out_csv_path) or "."
    ensure_dir(out_dir)
    df.to_csv(out_csv_path, index=False)

    print(f"[master] Saved: {out_csv_path}  (rows={len(df)})")
    return df




# =============================================================================
# Final paper figure assembly (PDF only)
# =============================================================================

FIGURE_FILES = {
    'lambda_main': 'Figure3.pdf',
    's_main': 'Figure4.pdf',
    'psi_main': 'Figure5.pdf',
    'delta_main': 'Figure6.pdf',
    'mubar_main': 'Figure7.pdf',
    'cwoff_app': 'FigureA1.pdf',
    's_high': 'FigureA2.pdf',
    'delta_price': 'FigureA3.pdf',
    'lambdaE_app': 'FigureA4.pdf',
    'T_app': 'FigureA5.pdf',
    'psi_high': 'FigureA6.pdf',
    'mubar_price': 'FigureA7.pdf',
}


def final_fig_path(fig_dir: str, key: str) -> str:
    ensure_dir(fig_dir)
    return os.path.join(fig_dir, FIGURE_FILES[key])


def _annot_text(region, p_off, p_on):
    return f"{region}"


MODEL_LABELS = {
    'Ch4': 'Fixed offline-tariff regime',
    'Ch5': 'Dual-tariff regime',
}

def _region_run_mid_indices(sub: pd.DataFrame):
    regions = list(sub['region'])
    if not regions:
        return []
    mids = []
    start = 0
    for i in range(1, len(regions) + 1):
        if i == len(regions) or regions[i] != regions[start]:
            mids.append((start + i - 1) // 2)
            start = i
    return mids

def _norm_title(title: str) -> str:
    t = str(title)
    t = t.replace(r'$\Lambda$', 'Λ').replace(r'$\psi$', 'ψ').replace(r'$\bar{\mu}$', 'μ̄')
    t = t.replace('$', '').replace('{', '').replace('}', '').replace('\\', '')
    return ' '.join(t.lower().split())

def _share_style_key(title: str, df: pd.DataFrame) -> str:
    t = _norm_title(title)
    ymax = float(df['mu_frac_on'].max()) if 'mu_frac_on' in df.columns and len(df) else 0.0
    if 'online share vs λ' in t or 'online share vs lambda' in t:
        return 'lambda_main'
    if 'online share vs s' in t:
        return 's_main' if ymax > 0.3 else 's_high'
    if 'online share vs ψ' in t or 'online share vs psi' in t:
        return 'psi_main' if ymax > 0.3 else 'psi_high'
    if 'moderate demand' in t:
        return 'mubar_572'
    if 'high demand' in t:
        return 'mubar_878'
    if 'online share, λ=572' in t or 'online share, lambda=572' in t:
        return 'share_572'
    if 'online share, λ=878' in t or 'online share, lambda=878' in t:
        return 'share_878'
    return 'generic'


_SHARE_LEGEND_KW = {
    # Figure 3
    'lambda_main': {'loc': 'upper right'},

    # Figure 4 first panel
    's_main': {'loc': 'upper right'},

    # Figure A2 first panel
    's_high': {'loc': 'center left', 'bbox_to_anchor': (0.02, 0.50)},

    # Figure 5
    'psi_main': {'loc': 'lower right'},

    # Figure A6 first panel
    'psi_high': {'loc': 'center left', 'bbox_to_anchor': (0.02, 0.50)},

    # Moderate / high demand share panels
    'mubar_572': {'loc': 'center left', 'bbox_to_anchor': (0.02, 0.50)},
    'mubar_878': {'loc': 'center left', 'bbox_to_anchor': (0.02, 0.50)},

    # A1 / A4 / A5 left-column share panels
    'share_572': {'loc': 'center left', 'bbox_to_anchor': (0.02, 0.50)},
    'share_878': {'loc': 'center left', 'bbox_to_anchor': (0.02, 0.50)},

    'generic': {'loc': 'upper left'},
}

# Keep data labels close to the actual points using small point offsets,
# instead of pushing them to faraway fixed axes coordinates.
_SHARE_LABEL_OFFSET_PT = {
    'lambda_main': {
        ('Ch4', 'V'):   (8, 8),
        ('Ch5', 'F'):   (8, -10),
        ('Ch4', 'BVF'): (-12, 8),
        ('Ch4', 'BF'):  (-12, -10),
        ('Ch5', 'BF'):  (-12, -10),
    },
    's_main': {
        ('Ch4', 'V'):   (-10, 8),
        ('Ch5', 'V'):   (8, 8),
        ('Ch5', 'F'):   (8, -10),
        ('Ch4', 'BF'):  (8, -10),
    },
    's_high': {
        ('Ch4', 'BVF'): (-12, 8),
        ('Ch5', 'BF'):  (-12, -10),
        ('Ch4', 'V'):   (8, 8),
        ('Ch5', 'F'):   (8, -10),
    },
    'psi_main': {
        ('Ch4', 'V'):   (8, 8),
        ('Ch5', 'F'):   (8, -10),
        ('Ch5', 'V'):   (8, 8),
    },
    'psi_high': {
        ('Ch4', 'V'):   (8, 8),
        ('Ch5', 'F'):   (8, -10),
        ('Ch5', 'V'):   (8, 8),
        ('Ch4', 'BVF'): (-12, 8),
        ('Ch5', 'BF'):  (-12, -10),
    },
    'mubar_572': {
        ('Ch4', 'BVF'): (-14, 8),
        ('Ch5', 'F'):   (10, -10),
        ('Ch4', 'V'):   (8, 8),
        ('Ch5', 'V'):   (8, 8),
        ('Ch4', 'BF'):  (-14, -10),
        ('Ch5', 'BF'):  (-14, -10),
    },
    'mubar_878': {
        ('Ch4', 'BVF'): (-12, 8),
        ('Ch5', 'BF'):  (-12, -10),
        ('Ch4', 'V'):   (8, 8),
        ('Ch5', 'F'):   (8, -10),
    },
    'share_572': {
        ('Ch4', 'V'):   (8, 8),
        ('Ch5', 'F'):   (8, -10),
        ('Ch4', 'BVF'): (-12, 8),
        ('Ch5', 'BF'):  (-12, -10),
    },
    'share_878': {
        ('Ch4', 'BVF'): (-12, 8),
        ('Ch5', 'BF'):  (-12, -10),
        ('Ch4', 'V'):   (8, 8),
        ('Ch5', 'F'):   (8, -10),
    },
}

def _fallback_label_offset_pt(model: str):
    if model == 'Ch4':
        return (8, 8)
    return (8, -10)


def _offset_alignment(dx: float, dy: float):
    ha = 'left' if dx >= 0 else 'right'
    if dy > 2:
        va = 'bottom'
    elif dy < -2:
        va = 'top'
    else:
        va = 'center'
    return ha, va


def _smart_label_offset(ax, x: float, y: float, dx: float, dy: float):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xr = max(xmax - xmin, 1e-9)
    yr = max(ymax - ymin, 1e-9)

    xrel = (x - xmin) / xr
    yrel = (y - ymin) / yr

    dx = 6 if dx >= 0 else -6
    dy = 6 if dy >= 0 else -6

    if xrel >= 0.90:
        dx = -8
    elif xrel <= 0.10:
        dx = 8

    if yrel >= 0.90:
        dy = -8
    elif yrel <= 0.10:
        dy = 8

    return dx, dy


def _disable_axis_offsets(ax):
    for axis in ('x', 'y'):
        try:
            ax.ticklabel_format(axis=axis, style='plain', useOffset=False)
        except (AttributeError, ValueError):
            pass
    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_visible(False)


def _format_price_axis(ax, values=None):
    _disable_axis_offsets(ax)
    finite = []
    if values is not None:
        for value in values:
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                finite.append(v)
    if finite and max(finite) - min(finite) < 1e-3:
        center = round(float(statistics.fmean(finite)), 1)
        ax.set_ylim(center - 1.0, center + 1.0)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.yaxis.get_offset_text().set_visible(False)


def _plot_capacity_panel(ax, df: pd.DataFrame, xcol: str, xlabel: str, title: str, annotate=True, style_key: Optional[str] = None):
    plotted = []
    for m in ['Ch4', 'Ch5']:
        sub = df[(df['model'] == m) & df['mu_frac_on'].notna()].sort_values(xcol)
        if len(sub) == 0:
            continue
        sub = sub.drop_duplicates(subset=xcol, keep='last').reset_index(drop=True)
        ax.plot(
            sub[xcol],
            sub['mu_frac_on'],
            marker='o',
            label=MODEL_LABELS.get(m, m),
        )
        plotted.append((m, sub))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$\mu_v/\mu_{\mathrm{total}}$')
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.margins(x=0.06)
    _disable_axis_offsets(ax)

    style_key = style_key or _share_style_key(title, df)
    legend_kw = dict(_SHARE_LEGEND_KW.get(style_key, {'loc': 'upper left'}))
    legend_kw.update({'fontsize': 9, 'framealpha': 0.9})
    ax.legend(**legend_kw)

    if not annotate:
        return

    manual_offsets = _SHARE_LABEL_OFFSET_PT.get(style_key, {})
    seen_keys = set()

    for m, sub in plotted:
        if 'region' not in sub.columns:
            continue
        label_idx = _region_run_mid_indices(sub)
        for idx in label_idx:
            row = sub.iloc[idx]
            region = row.get('region')
            if region is None or pd.isna(row.get('mu_frac_on')):
                continue
            key = (m, str(region))
            if key in seen_keys:
                continue
            seen_keys.add(key)

            dx, dy = manual_offsets.get(key, _fallback_label_offset_pt(m))
            dx, dy = _smart_label_offset(ax, float(row[xcol]), float(row['mu_frac_on']), dx, dy)
            ha, va = _offset_alignment(dx, dy)

            ax.annotate(
                _annot_text(region, row.get('p_off'), row.get('p_on')),
                xy=(float(row[xcol]), float(row['mu_frac_on'])),
                xytext=(dx, dy),
                textcoords='offset points',
                fontsize=10,
                ha=ha,
                va=va,
                bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none', alpha=0.72),
                clip_on=False,
                zorder=5,
            )


def _plot_ch4_price_panel(ax, df: pd.DataFrame, xcol: str, xlabel: str, title: str):
    price_values = []
    sub = df[df['p_on'].notna()].sort_values(xcol)
    if len(sub) > 0:
        sub = sub.drop_duplicates(subset=xcol, keep='last')
        price_values = list(sub['p_on'])
        ax.plot(
            sub[xcol], sub['p_on'],
            color=CH4_PRICE_COLOR,
            marker=CH4_PRICE_MARKER,
            linewidth=PRICE_LINEWIDTH,
            markersize=PRICE_MARKERSIZE,
            label=r'$p_{\mathrm{on}}$',
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Price')
    ax.set_title(title)
    ax.grid(True)
    _format_price_axis(ax, price_values)
    ax.legend()


def _plot_ch5_price_panel(ax, df: pd.DataFrame, xcol: str, xlabel: str, title: str):
    price_values = []
    sub = df[df['p_on'].notna()].sort_values(xcol)
    if len(sub) > 0:
        sub = sub.drop_duplicates(subset=xcol, keep='last')
        price_values = list(sub['p_on']) + list(sub['p_off'])
        ax.plot(
            sub[xcol], sub['p_on'],
            color=CH5_PON_COLOR,
            marker=CH5_PRICE_MARKER,
            linewidth=PRICE_LINEWIDTH,
            markersize=PRICE_MARKERSIZE,
            label=r'$p_{\mathrm{on}}$',
        )
        ax.plot(
            sub[xcol], sub['p_off'],
            color=CH5_POFF_COLOR,
            marker=CH5_PRICE_MARKER,
            linewidth=PRICE_LINEWIDTH,
            markersize=PRICE_MARKERSIZE,
            label=r'$p_{\mathrm{off}}$',
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Price')
    ax.set_title(title)
    ax.grid(True)
    _format_price_axis(ax, price_values)
    ax.legend()


def _save_final(fig, fig_dir: str, key: str):
    fig.savefig(final_fig_path(fig_dir, key), bbox_inches='tight', format='pdf')
    plt.close(fig)


def make_final_figures(fig_dir='figs', csv_dir='csv'):
    # load helper
    def readcsv(*parts):
        return pd.read_csv(os.path.join(csv_dir, *parts))

    # Figure 3
    cap = readcsv('lambda_capacity_share.csv')
    ch4 = readcsv('lambda_ch4_price.csv')
    ch5 = readcsv('lambda_ch5_price.csv')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    _plot_capacity_panel(axes[0], cap, 'Lambda', r'$\Lambda$', r'Online share vs $\Lambda$', style_key='lambda_main')
    _plot_ch4_price_panel(axes[1], ch4, 'Lambda', r'$\Lambda$', r'Fixed offline-tariff regime price vs $\Lambda$')
    _plot_ch5_price_panel(axes[2], ch5, 'Lambda', r'$\Lambda$', r'Dual-tariff regime prices vs $\Lambda$')
    fig.tight_layout(); _save_final(fig, fig_dir, 'lambda_main')

    # Figure 4
    cap = readcsv('LAMBDA_572', 's_capacity_share.csv')
    ch4 = readcsv('LAMBDA_572', 's_ch4_price.csv')
    ch5 = readcsv('LAMBDA_572', 's_ch5_price.csv')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    _plot_capacity_panel(axes[0], cap, 's', r'$s$', 'Online share vs $s$', style_key='s_main')
    _plot_ch4_price_panel(axes[1], ch4, 's', r'$s$', 'Fixed offline-tariff regime price vs $s$')
    _plot_ch5_price_panel(axes[2], ch5, 's', r'$s$', 'Dual-tariff regime prices vs $s$')
    fig.tight_layout(); _save_final(fig, fig_dir, 's_main')

    # Figure 5
    cap = readcsv('LAMBDA_572', 'psi_on_capacity_share.csv')
    ch4 = readcsv('LAMBDA_572', 'psi_on_ch4_price.csv')
    ch5 = readcsv('LAMBDA_572', 'psi_on_ch5_price.csv')
    cap = cap.rename(columns={'psi_on': 'psi'})
    ch4 = ch4.rename(columns={'psi_on': 'psi'})
    ch5 = ch5.rename(columns={'psi_on': 'psi'})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    _plot_capacity_panel(axes[0], cap, 'psi', r'$\psi$', r'Online share vs $\psi$', style_key='psi_main')
    _plot_ch4_price_panel(axes[1], ch4, 'psi', r'$\psi$', r'Fixed offline-tariff regime price vs $\psi$')
    _plot_ch5_price_panel(axes[2], ch5, 'psi', r'$\psi$', r'Dual-tariff regime prices vs $\psi$')
    fig.tight_layout(); _save_final(fig, fig_dir, 'psi_main')

    # Figure 6
    cap572 = readcsv('LAMBDA_572', 'delta_capacity_share.csv')
    cap878 = readcsv('LAMBDA_878', 'delta_capacity_share.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    delta_label_offsets = {
        'mubar_572': {
            ('Ch4', 'V'): (0, 8),
            ('Ch4', 'BVF'): (0, 8),
            ('Ch5', 'F'): (0, -8),
            ('Ch5', 'BF'): (0, -8),
        },
        'mubar_878': {
            ('Ch4', 'V'): (0, 8),
            ('Ch4', 'BVF'): (0, 8),
            ('Ch5', 'F'): (0, -8),
            ('Ch5', 'BF'): (0, -8),
        },
    }
    for ax, df, title, style_key in zip(
        axes,
        [cap572, cap878],
        [r'Moderate demand ($\Lambda=572$)', r'High demand ($\Lambda=878$)'],
        ['mubar_572', 'mubar_878'],
    ):
        labels = list(df['pair'].drop_duplicates())
        xs = np.arange(len(labels))
        for m in ['Ch4', 'Ch5']:
            sub = df[(df['model'] == m) & df['mu_frac_on'].notna()]
            y = [float(sub[sub['pair'] == lab]['mu_frac_on'].values[0]) if lab in sub['pair'].values else np.nan for lab in labels]
            ax.plot(xs, y, marker='o', label=MODEL_LABELS.get(m, m))
            for i, lab in enumerate(labels):
                hit = sub[sub['pair'] == lab]
                if len(hit) > 0:
                    row = hit.iloc[0]
                    base_dx, base_dy = delta_label_offsets.get(style_key, {}).get((m, row.get('region')), (0, 8 if m == 'Ch4' else -8))
                    dx, dy = _smart_label_offset(ax, float(xs[i]), float(y[i]), base_dx, base_dy)
                    ha, va = _offset_alignment(dx, dy)
                    ax.annotate(
                        _annot_text(row.get('region'), row.get('p_off'), row.get('p_on')),
                        (xs[i], y[i]), textcoords='offset points', xytext=(dx, dy),
                        fontsize=9, ha=ha, va=va,
                        bbox=dict(boxstyle='round,pad=0.10', fc='white', ec='none', alpha=0.70),
                        clip_on=False,
                    )
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_xlabel(r'$(\delta_1,\delta_2)$'); ax.set_ylabel(r'$\mu_v/\mu_{\mathrm{total}}$')
        ax.set_title(title); ax.grid(True, alpha=0.35)
        _disable_axis_offsets(ax)
        legend_kw = dict(_SHARE_LEGEND_KW.get(style_key, {'loc': 'upper left'}))
        legend_kw.update({'fontsize': 9, 'framealpha': 0.9})
        ax.legend(**legend_kw)
    fig.tight_layout(); _save_final(fig, fig_dir, 'delta_main')

    # Figure 7
    cap572 = readcsv('LAMBDA_572', 'mu_bar_L572_capacity_share.csv')
    cap878 = readcsv('LAMBDA_878', 'mu_bar_L878_capacity_share.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _plot_capacity_panel(axes[0], cap572, 'mu_bar', r'$\mu_{\mathrm{total}}$', r'Moderate demand ($\Lambda=572$)', style_key='mubar_572')
    _plot_capacity_panel(axes[1], cap878, 'mu_bar', r'$\mu_{\mathrm{total}}$', r'High demand ($\Lambda=878$)', style_key='mubar_878')
    fig.tight_layout(); _save_final(fig, fig_dir, 'mubar_main')

    # Figure A1
    cap572 = readcsv('LAMBDA_572', f'cw_off_given_on_{int(CW_ON_FIXED_FOR_OFF_SWEEP)}_capacity_share.csv')
    ch4572 = readcsv('LAMBDA_572', f'cw_off_given_on_{int(CW_ON_FIXED_FOR_OFF_SWEEP)}_ch4_price.csv')
    ch5572 = readcsv('LAMBDA_572', f'cw_off_given_on_{int(CW_ON_FIXED_FOR_OFF_SWEEP)}_ch5_price.csv')
    cap878 = readcsv('LAMBDA_878', f'cw_off_given_on_{int(CW_ON_FIXED_FOR_OFF_SWEEP)}_capacity_share.csv')
    ch4878 = readcsv('LAMBDA_878', f'cw_off_given_on_{int(CW_ON_FIXED_FOR_OFF_SWEEP)}_ch4_price.csv')
    ch5878 = readcsv('LAMBDA_878', f'cw_off_given_on_{int(CW_ON_FIXED_FOR_OFF_SWEEP)}_ch5_price.csv')
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    _plot_capacity_panel(axes[0,0], cap572, 'Cw_off', r'$C_{W,\mathrm{off}}$', r'Online share, $\Lambda=572$', style_key='share_572')
    _plot_ch4_price_panel(axes[0,1], ch4572, 'Cw_off', r'$C_{W,\mathrm{off}}$', r'Fixed offline-tariff regime, $\Lambda=572$')
    _plot_ch5_price_panel(axes[0,2], ch5572, 'Cw_off', r'$C_{W,\mathrm{off}}$', r'Dual-tariff regime, $\Lambda=572$')
    _plot_capacity_panel(axes[1,0], cap878, 'Cw_off', r'$C_{W,\mathrm{off}}$', r'Online share, $\Lambda=878$', style_key='share_878')
    _plot_ch4_price_panel(axes[1,1], ch4878, 'Cw_off', r'$C_{W,\mathrm{off}}$', r'Fixed offline-tariff regime, $\Lambda=878$')
    _plot_ch5_price_panel(axes[1,2], ch5878, 'Cw_off', r'$C_{W,\mathrm{off}}$', r'Dual-tariff regime, $\Lambda=878$')
    fig.tight_layout(); _save_final(fig, fig_dir, 'cwoff_app')

    # Figure A2
    cap = readcsv('LAMBDA_878', 's_capacity_share.csv')
    ch4 = readcsv('LAMBDA_878', 's_ch4_price.csv')
    ch5 = readcsv('LAMBDA_878', 's_ch5_price.csv')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    _plot_capacity_panel(axes[0], cap, 's', r'$s$', 'Online share vs $s$', style_key='s_high')
    _plot_ch4_price_panel(axes[1], ch4, 's', r'$s$', 'Fixed offline-tariff regime price vs $s$')
    _plot_ch5_price_panel(axes[2], ch5, 's', r'$s$', 'Dual-tariff regime prices vs $s$')
    fig.tight_layout(); _save_final(fig, fig_dir, 's_high')

    # Figure A3
    ch4572 = readcsv('LAMBDA_572', 'delta_ch4_price.csv')
    ch5572 = readcsv('LAMBDA_572', 'delta_ch5_price.csv')
    ch4878 = readcsv('LAMBDA_878', 'delta_ch4_price.csv')
    ch5878 = readcsv('LAMBDA_878', 'delta_ch5_price.csv')
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, df, title, is_ch5 in [(axes[0], ch4572, r'Fixed offline-tariff regime, $\Lambda=572$', False), (axes[1], ch5572, r'Dual-tariff regime, $\Lambda=572$', True), (axes[2], ch4878, r'Fixed offline-tariff regime, $\Lambda=878$', False), (axes[3], ch5878, r'Dual-tariff regime, $\Lambda=878$', True)]:
        labels = list(df['pair'].drop_duplicates())
        xs = np.arange(len(labels))
        sub = df.sort_values('pair')
        price_values = []
        if is_ch5:
            price_values = (
                [float(sub[sub['pair'] == lab]['p_on'].values[0]) for lab in labels]
                + [float(sub[sub['pair'] == lab]['p_off'].values[0]) for lab in labels]
            )
            ax.plot(
                xs,
                [float(sub[sub['pair'] == lab]['p_on'].values[0]) for lab in labels],
                color=CH5_PON_COLOR,
                marker=CH5_PRICE_MARKER,
                linewidth=PRICE_LINEWIDTH,
                markersize=PRICE_MARKERSIZE,
                label=r'$p_{\mathrm{on}}$',
            )
            ax.plot(
                xs,
                [float(sub[sub['pair'] == lab]['p_off'].values[0]) for lab in labels],
                color=CH5_POFF_COLOR,
                marker=CH5_PRICE_MARKER,
                linewidth=PRICE_LINEWIDTH,
                markersize=PRICE_MARKERSIZE,
                label=r'$p_{\mathrm{off}}$',
            )
        else:
            price_values = [float(sub[sub['pair'] == lab]['p_on'].values[0]) for lab in labels]
            ax.plot(
                xs,
                [float(sub[sub['pair'] == lab]['p_on'].values[0]) for lab in labels],
                color=CH4_PRICE_COLOR,
                marker=CH4_PRICE_MARKER,
                linewidth=PRICE_LINEWIDTH,
                markersize=PRICE_MARKERSIZE,
                label=r'$p_{\mathrm{on}}$',
            )
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_xlabel(r'$(\delta_1,\delta_2)$'); ax.set_ylabel('Price'); ax.set_title(title)
        ax.grid(True); _format_price_axis(ax, price_values); ax.legend()
    fig.tight_layout(); _save_final(fig, fig_dir, 'delta_price')

    # Figure A4
    cap572 = readcsv('LAMBDA_572', 'lambdaE_capacity_share.csv')
    ch4572 = readcsv('LAMBDA_572', 'lambdaE_ch4_price.csv')
    ch5572 = readcsv('LAMBDA_572', 'lambdaE_ch5_price.csv')
    cap878 = readcsv('LAMBDA_878', 'lambdaE_capacity_share.csv')
    ch4878 = readcsv('LAMBDA_878', 'lambdaE_ch4_price.csv')
    ch5878 = readcsv('LAMBDA_878', 'lambdaE_ch5_price.csv')
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    _plot_capacity_panel(axes[0,0], cap572, 'lambda_E', r'$\lambda_E$', r'Online share, $\Lambda=572$', style_key='share_572')
    _plot_ch4_price_panel(axes[0,1], ch4572, 'lambda_E', r'$\lambda_E$', r'Fixed offline-tariff regime, $\Lambda=572$')
    _plot_ch5_price_panel(axes[0,2], ch5572, 'lambda_E', r'$\lambda_E$', r'Dual-tariff regime, $\Lambda=572$')
    _plot_capacity_panel(axes[1,0], cap878, 'lambda_E', r'$\lambda_E$', r'Online share, $\Lambda=878$', style_key='share_878')
    _plot_ch4_price_panel(axes[1,1], ch4878, 'lambda_E', r'$\lambda_E$', r'Fixed offline-tariff regime, $\Lambda=878$')
    _plot_ch5_price_panel(axes[1,2], ch5878, 'lambda_E', r'$\lambda_E$', r'Dual-tariff regime, $\Lambda=878$')
    fig.tight_layout(); _save_final(fig, fig_dir, 'lambdaE_app')

    # Figure A5
    cap572 = readcsv('LAMBDA_572', 'T_capacity_share.csv')
    ch4572 = readcsv('LAMBDA_572', 'T_ch4_price.csv')
    ch5572 = readcsv('LAMBDA_572', 'T_ch5_price.csv')
    cap878 = readcsv('LAMBDA_878', 'T_capacity_share.csv')
    ch4878 = readcsv('LAMBDA_878', 'T_ch4_price.csv')
    ch5878 = readcsv('LAMBDA_878', 'T_ch5_price.csv')
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    _plot_capacity_panel(axes[0,0], cap572, 'T', r'$T$', r'Online share, $\Lambda=572$', style_key='share_572')
    _plot_ch4_price_panel(axes[0,1], ch4572, 'T', r'$T$', r'Fixed offline-tariff regime, $\Lambda=572$')
    _plot_ch5_price_panel(axes[0,2], ch5572, 'T', r'$T$', r'Dual-tariff regime, $\Lambda=572$')
    _plot_capacity_panel(axes[1,0], cap878, 'T', r'$T$', r'Online share, $\Lambda=878$', style_key='share_878')
    _plot_ch4_price_panel(axes[1,1], ch4878, 'T', r'$T$', r'Fixed offline-tariff regime, $\Lambda=878$')
    _plot_ch5_price_panel(axes[1,2], ch5878, 'T', r'$T$', r'Dual-tariff regime, $\Lambda=878$')
    fig.tight_layout(); _save_final(fig, fig_dir, 'T_app')

    # Figure A6
    cap = readcsv('LAMBDA_878', 'psi_on_capacity_share.csv').rename(columns={'psi_on': 'psi'})
    ch4 = readcsv('LAMBDA_878', 'psi_on_ch4_price.csv').rename(columns={'psi_on': 'psi'})
    ch5 = readcsv('LAMBDA_878', 'psi_on_ch5_price.csv').rename(columns={'psi_on': 'psi'})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    _plot_capacity_panel(axes[0], cap, 'psi', r'$\psi$', r'Online share vs $\psi$', style_key='psi_high')
    _plot_ch4_price_panel(axes[1], ch4, 'psi', r'$\psi$', r'Fixed offline-tariff regime price vs $\psi$')
    _plot_ch5_price_panel(axes[2], ch5, 'psi', r'$\psi$', r'Dual-tariff regime prices vs $\psi$')
    fig.tight_layout(); _save_final(fig, fig_dir, 'psi_high')

    # Figure A7
    ch4572 = readcsv('LAMBDA_572', 'mu_bar_L572_ch4_price.csv')
    ch5572 = readcsv('LAMBDA_572', 'mu_bar_L572_ch5_price.csv')
    ch4878 = readcsv('LAMBDA_878', 'mu_bar_L878_ch4_price.csv')
    ch5878 = readcsv('LAMBDA_878', 'mu_bar_L878_ch5_price.csv')
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    _plot_ch4_price_panel(axes[0], ch4572, 'mu_bar', r'$\mu_{\mathrm{total}}$', r'Fixed offline-tariff regime, $\Lambda=572$')
    _plot_ch5_price_panel(axes[1], ch5572, 'mu_bar', r'$\mu_{\mathrm{total}}$', r'Dual-tariff regime, $\Lambda=572$')
    _plot_ch4_price_panel(axes[2], ch4878, 'mu_bar', r'$\mu_{\mathrm{total}}$', r'Fixed offline-tariff regime, $\Lambda=878$')
    _plot_ch5_price_panel(axes[3], ch5878, 'mu_bar', r'$\mu_{\mathrm{total}}$', r'Dual-tariff regime, $\Lambda=878$')
    fig.tight_layout(); _save_final(fig, fig_dir, 'mubar_price')


if __name__ == '__main__':
    FIG_DIR = 'figs'
    CSV_DIR = 'csv'
    ensure_dir(FIG_DIR)
    ensure_dir(CSV_DIR)

    # save master table once
    _ = build_master_table_and_save(lambda_bases=LAMBDA_BASES, out_csv_path=os.path.join(CSV_DIR, 'master_sensitivity_results.csv'))

    # save parameter table
    params_rows = [
        dict(param='Lambda (strategic-patient arrival rate)', baseline=LAMBDA_BASES[0], grid=','.join(map(str, LAMBDA_GRID)), note=f'full sweeps at {LAMBDA_BASES}'),
        dict(param='Cw_off (offline waiting cost per hour)', baseline=CWOFF_BASE, grid=','.join(map(str, CW_OFF_GRID)), note=f'fixed Cw_on={CW_ON_FIXED_FOR_OFF_SWEEP} when sweeping Cw_off'),
        dict(param='s (offline extra utility)', baseline=BASE_COMMON['s'], grid=','.join(map(str, S_GRID)), note='diagnostic quality/equipment/trust'),
        dict(param='(delta1, delta2) referral probs', baseline=f"({BASE_CH4['delta1']},{BASE_CH4['delta2']})", grid='{' + ','.join(f'({d1},{d2})' for d1, d2 in DELTA_GRID) + '}', note='delta1 >= delta2 typical'),
        dict(param='lambda_E (offline-required arrival rate)', baseline=BASE_COMMON['lambda_E'], grid=','.join(map(str, LAM_E_GRID)), note='consumes offline capacity initially'),
        dict(param='T (transport/access cost)', baseline=BASE_COMMON['T'], grid=','.join(map(str, T_GRID)), note='affects both channels via (T - s) terms'),
        dict(param='p_off / caps', baseline='p_off=40 (Ch4), pbar_on=60 (Ch4); (60,80) caps in Ch5', grid='-', note='pricing rules'),
        dict(param='mu_total (total capacity per day)', baseline=MU_BAR_DEFAULT, grid=f"L=572 -> {MU_BAR_SWEEP_MAP[572.0]}; L=878 -> {MU_BAR_SWEEP_MAP[878.0]}", note='capacity sensitivity'),
        dict(param='Others', baseline=f"V={BASE_COMMON['V']}, r_balk={BASE_COMMON['r_balk']}", grid='-', note='shared settings'),
        dict(param='psi_on (online extra utility)', baseline=0, grid=','.join(map(str, PSI_ON_GRID)), note='patient-side utility bonus for online'),
    ]
    pd.DataFrame(params_rows).to_csv(os.path.join(CSV_DIR, 'sensitivity_params.csv'), index=False)

    # Write comparison outcomes used to discuss the revenue/accessibility tradeoff.
    _ = build_revenue_access_comparison(LAMBDA_GRID, os.path.join(CSV_DIR, 'revenue_access_comparison.csv'))

    # run raw sweeps (these will save raw PDFs into subfolders)
    sweep_Lambda(LAMBDA_GRID, CSV_DIR, os.path.join(FIG_DIR, 'raw_lambda'))
    for LAMBDA_BASE in LAMBDA_BASES:
        base_fig_dir = os.path.join(FIG_DIR, f'LAMBDA_{int(LAMBDA_BASE)}')
        base_csv_dir = os.path.join(CSV_DIR, f'LAMBDA_{int(LAMBDA_BASE)}')
        ensure_dir(base_fig_dir); ensure_dir(base_csv_dir)
        sweep_Cw_off_given_on(LAMBDA_BASE, Cw_on_fixed=CW_ON_FIXED_FOR_OFF_SWEEP, grid_off=CW_OFF_GRID, out_csv_dir=base_csv_dir, out_fig_dir=base_fig_dir)
        sweep_s(LAMBDA_BASE, S_GRID, base_csv_dir, base_fig_dir)
        sweep_delta_pairs(LAMBDA_BASE, DELTA_GRID, base_csv_dir, base_fig_dir)
        sweep_lambdaE(LAMBDA_BASE, LAM_E_GRID, base_csv_dir, base_fig_dir)
        sweep_T(LAMBDA_BASE, T_GRID, base_csv_dir, base_fig_dir)
        sweep_psi_on(LAMBDA_BASE, PSI_ON_GRID, base_csv_dir, base_fig_dir)
        if LAMBDA_BASE in MU_BAR_SWEEP_MAP:
            sweep_mu_bar_given_lambda(LAMBDA_BASE, MU_BAR_SWEEP_MAP[LAMBDA_BASE], out_csv_dir=base_csv_dir, out_fig_dir=base_fig_dir)

    # assemble final paper figures
    make_final_figures(FIG_DIR, CSV_DIR)
    print('Done. Final paper figures saved as PDF in ./figs')
