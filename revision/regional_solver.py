"""Support-wise optimizer for the continuous baseline provider problem.

The module leaves the Wardrop implementation in model.py unchanged. Prices are
eliminated exactly at fixed flow and capacity by enumerating the vertices of the
two-dimensional affine pricing problem. The provider problem is then solved on
support closures. F0 is already scalar; BF uses its square-root slack reduction;
BV and VF are nested bounded scalar problems.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import brentq, direct, minimize_scalar

from model import Model, evaluate


SUPPORTS = ("B", "V", "F", "BV", "BF", "VF")


def is_regional_baseline(model: Model) -> bool:
    """Return whether the model is covered by the regional characterization."""
    if model.servers or model.recurrent or model.elasticity != 0:
        return False
    if model.followup_fee != 1 or model.copay != 1 or model.revenue_E != 1:
        return False
    q, e, h, he, _v = model.matrices()
    return bool(np.allclose(h, q) and np.allclose(he, e) and np.allclose(e, q[:, 1]))


def _pricing_system(q: np.ndarray, g: np.ndarray, regime: str, model: Model, support: str):
    equalities: list[tuple[np.ndarray, float]] = []
    inequalities: list[tuple[np.ndarray, float]] = []

    if regime == "fixed":
        equalities.extend([
            (np.array([1.0, 0.0]), 60.0),
            (np.array([0.0, 1.0]), 40.0),
        ])
    elif regime == "online":
        equalities.append((np.array([0.0, 1.0]), 40.0))
        inequalities.extend([
            (np.array([-1.0, 0.0]), 0.0),
            (np.array([1.0, 0.0]), model.cap_on),
        ])
    elif regime == "dual":
        inequalities.extend([
            (np.array([-1.0, 0.0]), 0.0),
            (np.array([1.0, 0.0]), model.cap_on),
            (np.array([0.0, -1.0]), 0.0),
            (np.array([0.0, 1.0]), model.cap_off),
        ])
    else:
        raise ValueError(regime)

    qv, qf = q[:, 0], q[:, 1]
    if support == "B":
        inequalities.extend([(-qv, -g[0]), (-qf, -g[1])])
    elif support == "V":
        inequalities.extend([(qv, g[0]), (qv - qf, g[0] - g[1])])
    elif support == "F":
        inequalities.extend([(qf, g[1]), (qf - qv, g[1] - g[0])])
    elif support == "BV":
        equalities.append((qv, g[0]))
        inequalities.append((-qf, -g[1]))
    elif support == "BF":
        equalities.append((qf, g[1]))
        inequalities.append((-qv, -g[0]))
    elif support == "VF":
        equalities.append((qv - qf, g[0] - g[1]))
        inequalities.append((qv, g[0]))
    else:
        raise ValueError(support)
    return equalities, inequalities


def _best_vertex_price(
    loads: np.ndarray,
    q: np.ndarray,
    g: np.ndarray,
    regime: str,
    model: Model,
    support: str,
):
    """Solve the two-price LP by exhaustive vertex enumeration."""
    equalities, inequalities = _pricing_system(q, g, regime, model, support)
    eq_a = np.asarray([row for row, _rhs in equalities], dtype=float)
    eq_b = np.asarray([rhs for _row, rhs in equalities], dtype=float)
    rank = int(np.linalg.matrix_rank(eq_a)) if len(equalities) else 0
    candidates: list[np.ndarray] = []

    def solve_rows(rows, rhs):
        matrix = np.asarray(rows, dtype=float)
        if np.linalg.matrix_rank(matrix) < 2:
            return
        try:
            candidates.append(np.linalg.solve(matrix, np.asarray(rhs, dtype=float)))
        except np.linalg.LinAlgError:
            return

    if rank == 2:
        solution, *_ = np.linalg.lstsq(eq_a, eq_b, rcond=None)
        candidates.append(solution)
    elif rank == 1:
        base_index = next(i for i in range(len(eq_a)) if np.linalg.norm(eq_a[i]) > 0)
        for row, rhs in inequalities:
            solve_rows([eq_a[base_index], row], [eq_b[base_index], rhs])
    else:
        for (row_a, rhs_a), (row_b, rhs_b) in combinations(inequalities, 2):
            solve_rows([row_a, row_b], [rhs_a, rhs_b])

    feasible = []
    for price in candidates:
        if not np.all(np.isfinite(price)):
            continue
        if equalities and np.max(np.abs(eq_a @ price - eq_b)) > 2e-6:
            continue
        if inequalities and max(float(row @ price - rhs) for row, rhs in inequalities) > 2e-6:
            continue
        feasible.append(price)
    if not feasible:
        return None
    return max(feasible, key=lambda price: float(loads @ price))


def _closure_distance(support: str, target: np.ndarray, actual: np.ndarray, demand: float) -> float:
    """Distance relevant to the closure; adjacent boundary labels are allowed."""
    if support == "B":
        expected = np.zeros(2)
    elif support == "V":
        expected = np.array([demand, 0.0])
    elif support == "F":
        expected = np.array([0.0, demand])
    else:
        expected = target
    return float(np.max(np.abs(expected - actual)))


def _candidate(model: Model, regime: str, support: str, flow: np.ndarray, mu_v: float,
               required_slack: float = 1e-3):
    q, e, _h, _he, v = model.matrices()
    capacities = np.array([mu_v, model.capacity - mu_v])
    if np.min(capacities - model.required * e) < required_slack - 1e-9:
        return None
    loads = q @ flow + model.required * e
    slack = capacities - loads
    if np.min(slack) <= 1e-9:
        return None
    costs = np.array([model.cost_on, model.cost_off])
    g = v - q.T @ (costs / slack)
    prices = _best_vertex_price(loads, q, g, regime, model, support)
    if prices is None:
        return None
    checked = evaluate(model, (mu_v, float(prices[0]), float(prices[1])))
    if checked is None or checked["residual"] > 2e-5:
        return None
    actual = np.array([checked["flow_on"], checked["flow_off"]])
    if _closure_distance(support, flow, actual, model.demand) > 2e-3:
        return None
    result = dict(checked)
    result["regional_support"] = support
    result["regional_target_v"] = float(flow[0])
    result["regional_target_f"] = float(flow[1])
    return result


def _maximize_payload(function, lower: float, upper: float, maxfun: int, points=()):
    if not np.isfinite(lower + upper) or upper <= lower:
        return None
    cache: dict[float, dict | None] = {}

    def payload(value: float):
        value = float(np.clip(value, lower, upper))
        key = round(value, 12)
        if key not in cache:
            cache[key] = function(value)
        return cache[key]

    def loss(vector):
        item = payload(float(vector[0]))
        return 1e100 if item is None else -float(item["objective"])

    fit = direct(
        loss,
        [(lower, upper)],
        maxfun=maxfun,
        maxiter=maxfun,
        locally_biased=False,
        len_tol=1e-10,
        f_min_rtol=1e-12,
    )
    local = minimize_scalar(
        lambda value: loss([value]),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-11, "maxiter": 600},
    )
    trial_points = [float(fit.x[0]), float(local.x), lower, upper, *points]
    feasible = [payload(value) for value in trial_points if lower - 1e-9 <= value <= upper + 1e-9]
    feasible = [item for item in feasible if item is not None]
    if not feasible:
        return None
    best = max(feasible, key=lambda item: float(item["objective"]))
    best["regional_nfev"] = int(fit.nfev + local.nfev)
    return best


def _capacity_candidate(model: Model, regime: str, support: str, flow: np.ndarray,
                        maxfun: int, required_slack: float):
    q, e, *_ = model.matrices()
    loads = q @ flow + model.required * e
    lower = float(max(loads[0] + 1e-7, model.required * e[0] + required_slack))
    upper = float(min(model.capacity - loads[1] - 1e-7,
                      model.capacity - model.required * e[1] - required_slack))
    best = _maximize_payload(
        lambda mu: _candidate(model, regime, support, flow, mu, required_slack),
        lower,
        upper,
        maxfun,
    )
    if best is None or support not in ("V", "F"):
        return best
    center = float(best["mu_on"])
    span = upper - lower
    for scale in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9):
        radius = span * scale
        grid = np.linspace(max(lower, center - radius), min(upper, center + radius), 41)
        candidates = [_candidate(model, regime, support, flow, float(mu), required_slack) for mu in grid]
        candidates = [item for item in candidates if item is not None]
        if candidates:
            best = max([best, *candidates], key=lambda item: float(item["objective"]))
            center = float(best["mu_on"])
    return best


def _fixed_regime(model: Model, maxfun: int, required_slack: float):
    _q, e, *_ = model.matrices()
    lower = float(model.required * e[0] + required_slack)
    upper = float(model.capacity - model.required * e[1] - required_slack)

    def at_capacity(mu):
        result = evaluate(model, (mu, 60.0, 40.0))
        if result is None:
            return None
        result = dict(result)
        result["regional_support"] = result["region"]
        return result

    return _maximize_payload(at_capacity, lower, upper, maxfun)


def _bf_at_flow(model: Model, regime: str, admitted: float, required_slack: float):
    q, e, *_rest, v = model.matrices()
    qf = q[:, 1]
    if not np.allclose(e, qf):
        return None
    flow = np.array([0.0, admitted])
    loads = (admitted + model.required) * qf
    slack_total = model.capacity - float(np.sum(loads))
    if slack_total <= 2e-7:
        return None
    costs = np.array([model.cost_on, model.cost_off])
    weights = qf * costs
    if np.min(weights) <= 0:
        return None
    roots = np.sqrt(weights)
    slack_star = slack_total * roots[0] / float(np.sum(roots))

    def delay_cost(zv):
        return weights[0] / zv + weights[1] / (slack_total - zv)

    minimum_cost = delay_cost(slack_star)
    unconstrained_charge = v[1] - minimum_cost
    if regime == "fixed":
        target_charge = float(qf @ np.array([60.0, 40.0]))
    elif regime == "online":
        target_charge = min(unconstrained_charge, float(qf @ np.array([model.cap_on, 40.0])))
    elif regime == "dual":
        target_charge = min(unconstrained_charge, float(qf @ np.array([model.cap_on, model.cap_off])))
    else:
        raise ValueError(regime)
    target_cost = v[1] - target_charge
    if target_charge < -1e-8 or target_cost < minimum_cost - 1e-8:
        return None

    slack_candidates = [slack_star]
    if abs(target_cost - minimum_cost) <= 1e-7:
        pass
    else:
        left = brentq(lambda z: delay_cost(z) - target_cost, 1e-10, slack_star)
        right = brentq(lambda z: delay_cost(z) - target_cost, slack_star, slack_total - 1e-10)
        slack_candidates.extend([left, right])

    if regime == "online" and qf[0] > 0:
        qv = q[:, 0]

        def unused_utility(zv):
            slack = np.array([zv, slack_total - zv])
            charge_f = v[1] - qf @ (costs / slack)
            price_v = (charge_f - 40.0) / qf[0]
            prices = np.array([price_v, 40.0])
            return float(v[0] - qv @ prices - qv @ (costs / slack))

        grid = np.linspace(slack_total * 1e-8, slack_total * (1 - 1e-8), 161)
        values = [unused_utility(value) for value in grid]
        for left, right, value_left, value_right in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
            if value_left == 0:
                slack_candidates.append(float(left))
            elif value_left * value_right < 0:
                slack_candidates.append(brentq(unused_utility, float(left), float(right)))

        for price_v in (0.0, model.cap_on):
            charge = qf @ np.array([price_v, 40.0])
            cost = v[1] - charge
            if cost >= minimum_cost - 1e-9:
                if abs(cost - minimum_cost) <= 1e-8:
                    slack_candidates.append(slack_star)
                else:
                    slack_candidates.append(brentq(lambda z: delay_cost(z) - cost, 1e-10, slack_star))
                    slack_candidates.append(brentq(lambda z: delay_cost(z) - cost, slack_star, slack_total - 1e-10))

    candidates = []
    for slack_v in sorted(set(round(float(value), 12) for value in slack_candidates)):
        mu_v = float(loads[0] + slack_v)
        item = _candidate(model, regime, "BF", flow, mu_v, required_slack)
        if item is not None:
            candidates.append(item)
    return max(candidates, key=lambda item: item["objective"]) if candidates else None


def _bf_candidate(model: Model, regime: str, maxfun: int, required_slack: float):
    q, _e, *_rest, v = model.matrices()
    qf = q[:, 1]
    total_visits = float(np.sum(qf))
    max_admitted = min(
        model.demand,
        model.capacity / total_visits - model.required - 1e-8,
    )
    if max_admitted <= 0:
        return None
    costs = np.array([model.cost_on, model.cost_off])
    kf = float(np.sum(np.sqrt(qf * costs)) ** 2)
    points = [0.0, max_admitted]
    if v[1] + model.penalty > 0:
        total = (model.capacity - np.sqrt(kf * model.capacity / (v[1] + model.penalty))) / total_visits
        points.append(total - model.required)
    if regime == "fixed":
        cap_charge = float(qf @ np.array([60.0, 40.0]))
    elif regime == "online":
        cap_charge = float(qf @ np.array([model.cap_on, 40.0]))
    else:
        cap_charge = float(qf @ np.array([model.cap_on, model.cap_off]))
    if v[1] > cap_charge:
        total = (model.capacity - kf / (v[1] - cap_charge)) / total_visits
        points.append(total - model.required)
    return _maximize_payload(
        lambda admitted: _bf_at_flow(model, regime, admitted, required_slack),
        0.0,
        max_admitted,
        maxfun,
        points=points,
    )


def _nested_flow_candidate(model: Model, regime: str, support: str, inner: int,
                           outer: int, required_slack: float):
    def at_flow(value):
        if support == "BV":
            flow = np.array([value, 0.0])
        elif support == "BF":
            flow = np.array([0.0, value])
        elif support == "VF":
            flow = np.array([value, model.demand - value])
        else:
            raise ValueError(support)
        return _capacity_candidate(model, regime, support, flow, inner, required_slack)

    return _maximize_payload(at_flow, 0.0, model.demand, outer, points=(0.0, model.demand))


def optimize_regional(
    model: Model,
    regime: str = "dual",
    *,
    scalar_maxfun: int = 1600,
    inner_maxfun: int = 240,
    outer_maxfun: int = 480,
    required_slack: float = 1e-3,
):
    """Return the best reconstructed policy across baseline support closures."""
    if not is_regional_baseline(model):
        raise ValueError("Model is outside the continuous regional-baseline scope")
    if regime == "fixed":
        best = _fixed_regime(model, scalar_maxfun, required_slack)
        if best is None:
            raise ValueError("No feasible F0 policy")
        best["regime"] = regime
        best["solver"] = "regional"
        best["search_best"] = best["objective"]
        best["search_spread"] = 0.0
        return best

    candidates = []
    for support, flow in (
        ("B", np.zeros(2)),
        ("V", np.array([model.demand, 0.0])),
        ("F", np.array([0.0, model.demand])),
    ):
        item = _capacity_candidate(model, regime, support, flow, scalar_maxfun, required_slack)
        if item is not None:
            candidates.append(item)
    bf_closed = _bf_candidate(model, regime, scalar_maxfun, required_slack)
    if bf_closed is not None:
        candidates.append(bf_closed)
    for support in ("BV", "VF"):
        item = _nested_flow_candidate(model, regime, support, inner_maxfun, outer_maxfun, required_slack)
        if item is not None:
            candidates.append(item)
    if not candidates:
        raise ValueError("No feasible regional candidate")
    best = max(candidates, key=lambda item: float(item["objective"]))
    best["regime"] = regime
    best["solver"] = "regional"
    best["search_best"] = best["objective"]
    best["search_spread"] = 0.0
    best["regional_candidates"] = {
        item["regional_support"]: float(item["objective"]) for item in candidates
    }
    return best
