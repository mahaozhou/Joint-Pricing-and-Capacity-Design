from __future__ import annotations

"""
Discrete-event simulation validation for the analytical pricing-capacity policy.

The primary validation is model-consistent: for each tested policy, the
analytical Wardrop equilibrium is first recomputed, and online-eligible arrivals
are routed in the DES according to the resulting steady-state equilibrium
fractions. The DES benchmark is selected without inserting the analytical
policy into the simulation search; the analytical policy is evaluated only in
the final paired comparison.

The script also retains an optional out-of-model robustness test in which
arriving online-eligible patients observe the contemporaneous queue lengths and
choose the best service option at that state. Run it with --state-observing or
--all. That experiment should be interpreted as a stress test, not as the main
validation of the Wardrop calculation.
"""

import csv
import itertools
import math
import statistics
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import qmc

import price_style_modified_full_code_v2 as analytical


ROUTING_WARDROP = "wardrop"
ROUTING_STATE_OBSERVING = "state_observing"


@dataclass(frozen=True)
class GapInstance:
    name: str
    label: str
    chapter: int
    params: Dict[str, Any]


@dataclass(frozen=True)
class Policy:
    mu_on: float
    mu_off: float
    p_on: float
    p_off: float


@dataclass(frozen=True)
class SimSettings:
    search_horizon: float = 45.0
    search_warmup: float = 10.0
    search_reps: int = 2
    selection_horizon: float = 180.0
    selection_warmup: float = 40.0
    selection_reps: int = 6
    validation_horizon: float = 400.0
    validation_warmup: float = 80.0
    validation_reps: int = 12
    global_samples_ch4: int = 60
    global_samples_ch5: int = 96
    shortlist_size: int = 10
    local_anchor_count: int = 5
    base_seed_search: int = 202605240
    base_seed_selection: int = 202605241
    base_seed_validation: int = 202605242


def quick_settings() -> SimSettings:
    return SimSettings(
        search_horizon=18.0,
        search_warmup=4.0,
        search_reps=1,
        selection_horizon=45.0,
        selection_warmup=10.0,
        selection_reps=2,
        validation_horizon=80.0,
        validation_warmup=15.0,
        validation_reps=3,
        global_samples_ch4=8,
        global_samples_ch5=12,
        shortlist_size=4,
        local_anchor_count=2,
    )


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def stdev(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def normal95_ci(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    center = mean(values)
    half = 1.96 * stdev(values) / math.sqrt(len(values))
    return center - half, center + half


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def format_num(value: float, digits: int = 3) -> str:
    return "" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def make_param_object(instance: GapInstance):
    if instance.chapter == 4:
        return analytical.Ch4Params(**instance.params)
    return analytical.Ch5Params(**instance.params)


def solve_analytical(instance: GapInstance):
    params = make_param_object(instance)
    result = (
        analytical.ch4_solve_global(params)
        if instance.chapter == 4
        else analytical.ch5_solve_global(params)
    )
    return params, result


def as_policy(solution) -> Policy:
    return Policy(
        mu_on=float(solution.mu_on),
        mu_off=float(solution.mu_off),
        p_on=float(solution.p_on),
        p_off=float(solution.p_off),
    )


def policy_key(policy: Policy) -> Tuple[float, float, float]:
    return round(policy.mu_on, 6), round(policy.p_on, 6), round(policy.p_off, 6)


def stream_rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, stream]))


def utilities(p, policy: Policy, wait_on: float, wait_off: float) -> Tuple[float, float]:
    u_on = (
        p.V + getattr(p, "psi", 0.0) - policy.p_on - p.Cw_on * wait_on
        - p.delta1 * (p.T + policy.p_off + p.Cw_off * wait_off) + p.delta1 * p.s
    )
    u_off = (
        p.V + p.s - p.T - policy.p_off - p.Cw_off * wait_off
        - p.delta2 * (policy.p_on + p.Cw_on * wait_on)
    )
    return float(u_on), float(u_off)


def choose_state_observing_channel(
    p, policy: Policy, n_on: int, n_off: int, rng: np.random.Generator
) -> int:
    """Return 0=online, 1=offline, 2=balk under queue state observed at arrival."""
    wait_on = (n_on + 1.0) / policy.mu_on
    wait_off = (n_off + 1.0) / policy.mu_off
    u_on, u_off = utilities(p, policy, wait_on, wait_off)
    best_service = max(u_on, u_off)
    if best_service < -1e-10:
        return 2
    if abs(u_on - u_off) <= 1e-10:
        return int(rng.integers(0, 2))
    return 0 if u_on > u_off else 1


def wardrop_route_probabilities(p, equilibrium: Dict[str, Any]) -> Tuple[float, float, float]:
    p_on = max(0.0, min(1.0, float(equilibrium["lambda_on"]) / p.Lambda))
    p_off = max(0.0, min(1.0, float(equilibrium["lambda_off"]) / p.Lambda))
    p_balk = max(0.0, 1.0 - p_on - p_off)
    total = p_on + p_off + p_balk
    return p_on / total, p_off / total, p_balk / total


def choose_wardrop_channel(route_probs: Tuple[float, float, float], rng: np.random.Generator) -> int:
    draw = float(rng.random())
    if draw < route_probs[0]:
        return 0
    if draw < route_probs[0] + route_probs[1]:
        return 1
    return 2


def next_service_time(
    queue: Deque[bool], now: float, mu: float, rng: np.random.Generator
) -> float:
    return now + float(rng.exponential(1.0 / mu)) if queue else math.inf


def simulate_des_replication(
    p,
    policy: Policy,
    *,
    horizon: float,
    warmup: float,
    seed: int,
    routing_mode: str,
    equilibrium: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Simulate one two-channel event-driven queue replication."""
    if routing_mode == ROUTING_WARDROP and equilibrium is None:
        raise ValueError("Wardrop routing requires an equilibrium.")
    route_probs = (
        wardrop_route_probabilities(p, equilibrium)
        if routing_mode == ROUTING_WARDROP and equilibrium is not None
        else None
    )

    eligible_rng = stream_rng(seed, 1)
    required_rng = stream_rng(seed, 2)
    choice_rng = stream_rng(seed, 3)
    service_on_rng = stream_rng(seed, 4)
    service_off_rng = stream_rng(seed, 5)
    referral_on_rng = stream_rng(seed, 6)
    referral_off_rng = stream_rng(seed, 7)

    queue_on: Deque[bool] = deque()
    queue_off: Deque[bool] = deque()
    now = 0.0
    next_eligible = float(eligible_rng.exponential(1.0 / p.Lambda))
    next_required = float(required_rng.exponential(1.0 / p.lambda_E))
    next_on_service = math.inf
    next_off_service = math.inf

    revenue = eligible_arrivals = accepted = balked = 0.0
    completed_on = completed_off = 0.0
    number_area_on = number_area_off = 0.0

    while now < horizon:
        event_time = min(next_eligible, next_required, next_on_service, next_off_service, horizon)
        area_start = max(now, warmup)
        if event_time > area_start:
            number_area_on += len(queue_on) * (event_time - area_start)
            number_area_off += len(queue_off) * (event_time - area_start)
        now = event_time
        if now >= horizon:
            break
        record = now >= warmup

        if event_time == next_eligible:
            if routing_mode == ROUTING_WARDROP and route_probs is not None:
                channel = choose_wardrop_channel(route_probs, choice_rng)
            else:
                channel = choose_state_observing_channel(p, policy, len(queue_on), len(queue_off), choice_rng)
            if record:
                eligible_arrivals += 1.0
            if channel == 0:
                idle = not queue_on
                queue_on.append(True)
                if record:
                    accepted += 1.0
                if idle:
                    next_on_service = next_service_time(queue_on, now, policy.mu_on, service_on_rng)
            elif channel == 1:
                idle = not queue_off
                queue_off.append(True)
                if record:
                    accepted += 1.0
                if idle:
                    next_off_service = next_service_time(queue_off, now, policy.mu_off, service_off_rng)
            elif record:
                balked += 1.0
                revenue -= p.r_balk
            next_eligible = now + float(eligible_rng.exponential(1.0 / p.Lambda))

        elif event_time == next_required:
            idle = not queue_off
            queue_off.append(True)
            if idle:
                next_off_service = next_service_time(queue_off, now, policy.mu_off, service_off_rng)
            next_required = now + float(required_rng.exponential(1.0 / p.lambda_E))

        elif event_time == next_on_service:
            initial_encounter = queue_on.popleft()
            if record:
                completed_on += 1.0
                revenue += policy.p_on
            if initial_encounter and referral_on_rng.random() < p.delta1:
                idle = not queue_off
                queue_off.append(False)
                if idle:
                    next_off_service = next_service_time(queue_off, now, policy.mu_off, service_off_rng)
            next_on_service = next_service_time(queue_on, now, policy.mu_on, service_on_rng)

        else:
            initial_encounter = queue_off.popleft()
            if record:
                completed_off += 1.0
                revenue += policy.p_off
            if initial_encounter and referral_off_rng.random() < p.delta2:
                idle = not queue_on
                queue_on.append(False)
                if idle:
                    next_on_service = next_service_time(queue_on, now, policy.mu_on, service_on_rng)
            next_off_service = next_service_time(queue_off, now, policy.mu_off, service_off_rng)

    duration = max(horizon - warmup, 1e-12)
    return {
        "revenue_d": revenue / duration,
        "access_rate": accepted / max(eligible_arrivals, 1.0),
        "balk_rate": balked / max(eligible_arrivals, 1.0),
        "mean_number_on": number_area_on / duration,
        "mean_number_off": number_area_off / duration,
        "throughput_on": completed_on / duration,
        "throughput_off": completed_off / duration,
        "terminal_queue": float(len(queue_on) + len(queue_off)),
    }


def stage_configuration(settings: SimSettings, stage: str):
    if stage == "search":
        return settings.search_horizon, settings.search_warmup, settings.search_reps, settings.base_seed_search
    if stage == "selection":
        return settings.selection_horizon, settings.selection_warmup, settings.selection_reps, settings.base_seed_selection
    return settings.validation_horizon, settings.validation_warmup, settings.validation_reps, settings.base_seed_validation


def policy_equilibrium_for_mode(p, policy: Policy, routing_mode: str) -> Optional[Dict[str, Any]]:
    if routing_mode == ROUTING_WARDROP:
        return analytical.solve_policy_equilibrium(p, policy.mu_on, policy.p_on, policy.p_off)
    return None


def evaluate_policy(
    p,
    policy: Policy,
    *,
    settings: SimSettings,
    stage: str,
    routing_mode: str,
) -> List[Dict[str, float]]:
    if (
        policy.mu_on <= 0.0 or policy.mu_off <= 0.0
        or policy.p_on < 0.0 or policy.p_on > p.pbar_on + 1e-9
        or (
            isinstance(p, analytical.Ch5Params)
            and (policy.p_off < 0.0 or policy.p_off > p.pbar_off + 1e-9)
        )
    ):
        return [{"revenue_d": -1e12, "access_rate": 0.0, "balk_rate": 1.0}]

    equilibrium = policy_equilibrium_for_mode(p, policy, routing_mode)
    if routing_mode == ROUTING_WARDROP and equilibrium is None:
        return [{"revenue_d": -1e12, "access_rate": 0.0, "balk_rate": 1.0}]

    horizon, warmup, reps, base_seed = stage_configuration(settings, stage)
    return [
        simulate_des_replication(
            p,
            policy,
            horizon=horizon,
            warmup=warmup,
            seed=base_seed + rep,
            routing_mode=routing_mode,
            equilibrium=equilibrium,
        )
        for rep in range(reps)
    ]


def policy_from_unit_point(p, chapter: int, point: Sequence[float]) -> Policy:
    mu_on = 1.0 + float(point[0]) * (p.mu_bar - 2.0)
    p_on = float(point[1]) * p.pbar_on
    p_off = p.p_off if chapter == 4 else float(point[2]) * p.pbar_off
    return Policy(mu_on=mu_on, mu_off=p.mu_bar - mu_on, p_on=p_on, p_off=p_off)


def policy_from_vector(p, chapter: int, values: Sequence[float]) -> Policy:
    mu_on = clip(float(values[0]), 1.0, p.mu_bar - 1.0)
    p_on = clip(float(values[1]), 0.0, p.pbar_on)
    p_off = p.p_off if chapter == 4 else clip(float(values[2]), 0.0, p.pbar_off)
    return Policy(mu_on=mu_on, mu_off=p.mu_bar - mu_on, p_on=p_on, p_off=p_off)


def policy_to_vector(policy: Policy, chapter: int) -> Tuple[float, ...]:
    if chapter == 4:
        return policy.mu_on, policy.p_on
    return policy.mu_on, policy.p_on, policy.p_off


def structured_grid_policies(p, chapter: int) -> List[Policy]:
    mu_fracs = (0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60, 0.80, 0.95)
    p_on_fracs = (0.00, 0.25, 0.50, 0.75, 1.00)
    p_off_fracs = (0.25, 0.50, 0.65, 0.75, 0.85, 1.00)
    policies: List[Policy] = []
    for mu_frac in mu_fracs:
        for p_on_frac in p_on_fracs:
            if chapter == 4:
                policies.append(
                    Policy(
                        mu_on=1.0 + mu_frac * (p.mu_bar - 2.0),
                        mu_off=p.mu_bar - (1.0 + mu_frac * (p.mu_bar - 2.0)),
                        p_on=p_on_frac * p.pbar_on,
                        p_off=p.p_off,
                    )
                )
            else:
                for p_off_frac in p_off_fracs:
                    policies.append(
                        Policy(
                            mu_on=1.0 + mu_frac * (p.mu_bar - 2.0),
                            mu_off=p.mu_bar - (1.0 + mu_frac * (p.mu_bar - 2.0)),
                            p_on=p_on_frac * p.pbar_on,
                            p_off=p_off_frac * p.pbar_off,
                        )
                    )
    return unique_policies(policies)


def unique_policies(policies: Iterable[Policy]) -> List[Policy]:
    seen = set()
    output: List[Policy] = []
    for policy in policies:
        key = policy_key(policy)
        if key not in seen:
            seen.add(key)
            output.append(policy)
    return output


def local_refinement_policies(p, chapter: int, anchors: Sequence[Policy]) -> List[Policy]:
    candidates: List[Policy] = []
    for anchor in anchors:
        for dmu in (-20.0, -10.0, 0.0, 10.0, 20.0):
            for dpon in (-4.0, -2.0, 0.0, 2.0, 4.0):
                p_off_offsets = (0.0,) if chapter == 4 else (-4.0, -2.0, 0.0, 2.0, 4.0)
                for dpoff in p_off_offsets:
                    mu_on = clip(anchor.mu_on + dmu, 1.0, p.mu_bar - 1.0)
                    candidates.append(
                        Policy(
                            mu_on=mu_on,
                            mu_off=p.mu_bar - mu_on,
                            p_on=clip(anchor.p_on + dpon, 0.0, p.pbar_on),
                            p_off=p.p_off if chapter == 4 else clip(anchor.p_off + dpoff, 0.0, p.pbar_off),
                        )
                    )
    return unique_policies(candidates)


def pattern_refinement_policies(
    p,
    chapter: int,
    anchors: Sequence[Policy],
    screen_value,
) -> None:
    steps = (
        (160.0, 20.0),
        (80.0, 10.0),
        (40.0, 5.0),
        (20.0, 2.5),
        (10.0, 1.0),
        (5.0, 0.5),
    )
    dimension = 2 if chapter == 4 else 3
    for anchor in anchors:
        current = anchor
        current_value = screen_value(current)
        for mu_step, price_step in steps:
            improved = True
            passes = 0
            while improved and passes < 3:
                passes += 1
                improved = False
                best_policy = current
                best_value = current_value
                current_vector = policy_to_vector(current, chapter)
                for signs in itertools.product((-1.0, 0.0, 1.0), repeat=dimension):
                    offsets = [signs[0] * mu_step, signs[1] * price_step]
                    if chapter == 5:
                        offsets.append(signs[2] * price_step)
                    candidate = policy_from_vector(
                        p,
                        chapter,
                        [value + offset for value, offset in zip(current_vector, offsets)],
                    )
                    value = screen_value(candidate)
                    if value > best_value + 1e-6:
                        best_policy = candidate
                        best_value = value
                if policy_key(best_policy) != policy_key(current):
                    current = best_policy
                    current_value = best_value
                    improved = True


def identify_simulation_benchmark(
    instance: GapInstance,
    p,
    analytical_policy: Policy,
    settings: SimSettings,
    routing_mode: str,
) -> Tuple[Policy, List[Dict[str, Any]], int]:
    dimension = 2 if instance.chapter == 4 else 3
    n_samples = settings.global_samples_ch4 if instance.chapter == 4 else settings.global_samples_ch5
    sampler = qmc.LatinHypercube(d=dimension, seed=20260524 + instance.chapter + int(p.Lambda))
    policies = [policy_from_unit_point(p, instance.chapter, point) for point in sampler.random(n_samples)]
    policies.extend(structured_grid_policies(p, instance.chapter))
    archive: Dict[Tuple[float, float, float], Tuple[Policy, float]] = {}

    def screen(policy: Policy) -> float:
        if policy_key(policy) not in archive:
            rows = evaluate_policy(p, policy, settings=settings, stage="search", routing_mode=routing_mode)
            archive[policy_key(policy)] = (policy, mean([row["revenue_d"] for row in rows]))
        return archive[policy_key(policy)][1]

    for policy in unique_policies(policies):
        screen(policy)
    ranked = sorted(archive.values(), key=lambda item: item[1], reverse=True)
    anchors = [item[0] for item in ranked[: settings.local_anchor_count]]
    for policy in local_refinement_policies(p, instance.chapter, anchors):
        screen(policy)
    ranked = sorted(archive.values(), key=lambda item: item[1], reverse=True)
    pattern_refinement_policies(
        p,
        instance.chapter,
        [item[0] for item in ranked[: settings.local_anchor_count]],
        screen,
    )
    ranked = sorted(archive.values(), key=lambda item: item[1], reverse=True)
    shortlist = unique_policies([item[0] for item in ranked[: settings.shortlist_size]])

    selection_rows: List[Dict[str, Any]] = []
    for policy in shortlist:
        rows = evaluate_policy(p, policy, settings=settings, stage="selection", routing_mode=routing_mode)
        selection_rows.append({
            "routing_mode": routing_mode,
            "instance_name": instance.name,
            "policy_type": "des_search_match" if policy_key(policy) == policy_key(analytical_policy) else "des_search",
            "mu_on": policy.mu_on,
            "mu_off": policy.mu_off,
            "p_on": policy.p_on,
            "p_off": policy.p_off,
            "selection_revenue_mean": mean([row["revenue_d"] for row in rows]),
            "selection_revenue_std": stdev([row["revenue_d"] for row in rows]),
            "selection_access_rate": mean([row["access_rate"] for row in rows]),
            "selection_balk_rate": mean([row["balk_rate"] for row in rows]),
        })
    selection_rows.sort(key=lambda row: row["selection_revenue_mean"], reverse=True)
    best = selection_rows[0]
    return Policy(best["mu_on"], best["mu_off"], best["p_on"], best["p_off"]), selection_rows, len(archive)


def build_summary_row(
    instance: GapInstance,
    solution,
    p,
    analytical_policy: Policy,
    benchmark_policy: Policy,
    settings: SimSettings,
    policies_evaluated: int,
    routing_mode: str,
) -> Dict[str, Any]:
    analytical_rows = evaluate_policy(
        p, analytical_policy, settings=settings, stage="validation", routing_mode=routing_mode
    )
    benchmark_rows = evaluate_policy(
        p, benchmark_policy, settings=settings, stage="validation", routing_mode=routing_mode
    )
    ana_values = [row["revenue_d"] for row in analytical_rows]
    best_values = [row["revenue_d"] for row in benchmark_rows]
    gaps = [best - ana for best, ana in zip(best_values, ana_values)]
    gap = mean(gaps)
    gap_percent = gap / abs(mean(best_values)) * 100.0 if abs(mean(best_values)) > 1e-12 else float("nan")
    ci_low, ci_high = normal95_ci(gaps)
    interpretation = "Within 1%" if math.isfinite(gap_percent) and gap_percent <= 1.0 else "Material gap"
    if ci_high < 0.0:
        interpretation = "Analytical not worse"
    model_access = (solution.lambda_on + solution.lambda_off) / p.Lambda
    return {
        "routing_mode": routing_mode,
        "instance_name": instance.name,
        "instance_label": instance.label,
        "chapter": instance.chapter,
        "analytical_region": solution.region,
        "analytical_mu_on": analytical_policy.mu_on,
        "analytical_p_on": analytical_policy.p_on,
        "analytical_p_off": analytical_policy.p_off,
        "benchmark_mu_on": benchmark_policy.mu_on,
        "benchmark_p_on": benchmark_policy.p_on,
        "benchmark_p_off": benchmark_policy.p_off,
        "analytical_model_revenue": solution.revenue,
        "analytical_model_access_rate": model_access,
        "analytical_model_balk_rate": solution.lambda_balk / p.Lambda,
        "analytical_des_revenue": mean(ana_values),
        "best_des_revenue": mean(best_values),
        "gap_best_minus_analytical": gap,
        "gap_percent_of_best": gap_percent,
        "paired_gap_ci_low": ci_low,
        "paired_gap_ci_high": ci_high,
        "analytical_access_rate": mean([row["access_rate"] for row in analytical_rows]),
        "best_access_rate": mean([row["access_rate"] for row in benchmark_rows]),
        "analytical_balk_rate": mean([row["balk_rate"] for row in analytical_rows]),
        "best_balk_rate": mean([row["balk_rate"] for row in benchmark_rows]),
        "policies_evaluated": policies_evaluated,
        "interpretation": interpretation,
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_policy_tuple(mu_on: float, p_on: float, p_off: float) -> str:
    return f"({format_num(mu_on, 2)}, {format_num(p_on, 2)}, {format_num(p_off, 2)})"


def compact_instance_label(label: str) -> str:
    return label.replace("Fixed tariff,", "Fixed,").replace("Dual tariff,", "Dual,")


def build_latex_table(rows: Sequence[Dict[str, Any]], routing_mode: str) -> str:
    if routing_mode == ROUTING_WARDROP:
        caption = "Independent model-consistent DES benchmark comparison."
        label = "tab:model-consistent-simulation-gap"
    else:
        caption = "Robustness test under state-observing patient choice in the discrete-event simulation environment."
        label = "tab:state-observing-simulation-gap"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
    ]
    if routing_mode == ROUTING_WARDROP:
        lines.extend([
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2pt}",
            r"\renewcommand{\arraystretch}{1.10}",
            r"\begin{tabular}{@{}lccrrc@{}}",
            r"\toprule",
            r"Instance & $\pi^A$ & $\pi^D$ & $J^A$ & $J^D$ & $\Delta$ [95\% CI] \\",
            r"\midrule",
        ])
    else:
        lines.extend([
            r"\renewcommand{\arraystretch}{1.10}",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Instance & Analytical DES & DES-selected DES & Gap & 95\% CI & Gap (\%) \\",
            r"\midrule",
        ])
    for row in rows:
        digits = 3 if routing_mode == ROUTING_WARDROP else (
            1 if max(abs(row["paired_gap_ci_low"]), abs(row["paired_gap_ci_high"])) >= 100 else 2
        )
        ci = f"[{format_num(row['paired_gap_ci_low'], digits)}, {format_num(row['paired_gap_ci_high'], digits)}]"
        if routing_mode == ROUTING_WARDROP:
            analytical_policy = format_policy_tuple(
                row["analytical_mu_on"], row["analytical_p_on"], row["analytical_p_off"]
            )
            benchmark_policy = format_policy_tuple(
                row["benchmark_mu_on"], row["benchmark_p_on"], row["benchmark_p_off"]
            )
            lines.append(
                f"{compact_instance_label(row['instance_label'])} & {analytical_policy} & {benchmark_policy} & "
                f"{format_num(row['analytical_des_revenue'], 2)} & {format_num(row['best_des_revenue'], 2)} & "
                f"{format_num(row['gap_best_minus_analytical'], 3)} {ci} \\\\"
            )
        else:
            lines.append(
                f"{row['instance_label']} & {format_num(row['analytical_des_revenue'], 1)} & "
                f"{format_num(row['best_des_revenue'], 1)} & {format_num(row['gap_best_minus_analytical'], 1)} & "
                f"{ci} & {format_num(row['gap_percent_of_best'], 2)}\\% \\\\"
            )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def example_instances() -> List[GapInstance]:
    common = {
        "lambda_E": 30.0,
        "delta1": 0.35,
        "delta2": 0.10,
        "V": 50.0,
        "T": 20.0,
        "s": 30.0,
        "mu_bar": 972.0,
        "Cw_on": 112.0,
        "Cw_off": 168.0,
        "r_balk": 10.0,
        "M_const": 0.0,
        "outer_grid": 121,
        "refine_topk": 3,
        "pbar_on": 60.0,
        "psi": 0.0,
    }
    return [
        GapInstance("fixed_L572", r"Fixed tariff, $\Lambda=572$", 4, {**common, "Lambda": 572.0, "p_off": 40.0}),
        GapInstance("dual_L572", r"Dual tariff, $\Lambda=572$", 5, {**common, "Lambda": 572.0, "pbar_off": 80.0}),
        GapInstance("fixed_L878", r"Fixed tariff, $\Lambda=878$", 4, {**common, "Lambda": 878.0, "p_off": 40.0}),
        GapInstance("dual_L878", r"Dual tariff, $\Lambda=878$", 5, {**common, "Lambda": 878.0, "pbar_off": 80.0}),
    ]


def run_gap_study(
    instances: Sequence[GapInstance],
    *,
    outdir: Path,
    settings: SimSettings,
    routing_mode: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    summaries: List[Dict[str, Any]] = []
    selection_rows: List[Dict[str, Any]] = []
    for instance in instances:
        p, solution = solve_analytical(instance)
        analytical_policy = as_policy(solution)
        benchmark, candidates, evaluated = identify_simulation_benchmark(
            instance, p, analytical_policy, settings, routing_mode
        )
        selection_rows.extend(candidates)
        row = build_summary_row(
            instance,
            solution,
            p,
            analytical_policy,
            benchmark,
            settings,
            evaluated,
            routing_mode,
        )
        summaries.append(row)
        print(
            f"{routing_mode} | {instance.name}: analytical={row['analytical_des_revenue']:.3f}, "
            f"benchmark={row['best_des_revenue']:.3f}, gap={row['gap_best_minus_analytical']:.3f} "
            f"({row['gap_percent_of_best']:.3f}%), access="
            f"({row['analytical_access_rate']:.4f}, {row['best_access_rate']:.4f})"
        )
    latex = build_latex_table(summaries, routing_mode)
    write_csv(outdir / "des_gap_summary.csv", summaries)
    write_csv(outdir / "des_gap_selection.csv", selection_rows)
    (outdir / "des_gap_table.tex").write_text(latex, encoding="utf-8")
    return summaries, selection_rows, latex


def output_dir_for_mode(routing_mode: str, quick: bool) -> Path:
    prefix = "quick_" if quick else ""
    if routing_mode == ROUTING_WARDROP:
        return Path(f"{prefix}gap_outputs_model_consistent_des")
    return Path(f"{prefix}gap_outputs_state_observing_des")


def main() -> None:
    args = set(sys.argv[1:])
    quick = "--quick" in args
    settings = quick_settings() if quick else SimSettings()
    modes = [ROUTING_WARDROP]
    if "--state-observing" in args:
        modes = [ROUTING_STATE_OBSERVING]
    if "--all" in args:
        modes = [ROUTING_WARDROP, ROUTING_STATE_OBSERVING]

    for routing_mode in modes:
        outdir = output_dir_for_mode(routing_mode, quick)
        summaries, _, latex = run_gap_study(
            example_instances(), outdir=outdir, settings=settings, routing_mode=routing_mode
        )
        print(f"\n=== {routing_mode.upper()} DISCRETE-EVENT SIMULATION GAP SUMMARY ===")
        for row in summaries:
            print(
                row["instance_name"],
                "| model_rev=", format_num(row["analytical_model_revenue"]),
                "| des_rev=", format_num(row["analytical_des_revenue"]),
                "| gap=", format_num(row["gap_best_minus_analytical"]),
                "| gap_pct=", format_num(row["gap_percent_of_best"]),
                "| paired 95% CI=[", format_num(row["paired_gap_ci_low"]), ",",
                format_num(row["paired_gap_ci_high"]), "]",
                "|", row["interpretation"],
            )
        print("\n=== LATEX TABLE ===")
        print(latex)
        print(f"\nFiles written to: {outdir.resolve()}\n")


if __name__ == "__main__":
    main()
