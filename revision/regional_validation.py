"""Validate the regional baseline solver on all 84 primary configurations."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from experiments import design
from model import Model, optimize
from regional_solver import optimize_regional


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision"
CACHE = OUT / "regional_cache"


def primary_tasks():
    return [task for task in design() if task["group"] == "referral"]


def _compact(result):
    keys = (
        "p_on", "p_off", "mu_on", "mu_off", "flow_on", "flow_off",
        "objective", "gross", "penalty", "access", "region", "residual",
        "wait_on", "wait_off", "exposure",
    )
    return {key: result[key] for key in keys}


def run_one(task):
    signature = {
        "task": task,
        "regional": {"scalar": 600, "inner": 90, "outer": 150},
        "de": {"iterations": 450, "restarts": 8},
        "version": 3,
    }
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())

    model = Model(**task["model"])
    regional = optimize_regional(
        model,
        task["regime"],
        scalar_maxfun=600,
        inner_maxfun=90,
        outer_maxfun=150,
    )
    de = optimize(
        model,
        task["regime"],
        seed=7301,
        iterations=450,
        restarts=8,
        secondary=False,
    )
    initial_gap = float(regional["objective"] - de["objective"])
    strengthened = abs(initial_gap) > 1e-3
    if strengthened:
        regional = optimize_regional(
            model,
            task["regime"],
            scalar_maxfun=2200,
            inner_maxfun=320,
            outer_maxfun=650,
        )
        de = optimize(
            model,
            task["regime"],
            seed=19001,
            iterations=900,
            restarts=14,
            secondary=False,
        )
    gap = float(regional["objective"] - de["objective"])
    result = {
        "group": task["group"],
        "label": task["label"],
        "regime": task["regime"],
        "model": asdict(model),
        "regional": _compact(regional),
        "de": _compact(de),
        "objective_gap_regional_minus_de": gap,
        "objective_abs_gap": abs(gap),
        "objective_relative_gap": abs(gap) / max(1.0, abs(de["objective"])),
        "support_match": regional["region"] == de["region"],
        "initial_gap": initial_gap,
        "strengthened": strengthened,
        "key": key,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def summarize(rows):
    worst = max(rows, key=lambda row: row["objective_abs_gap"])
    return {
        "count": len(rows),
        "max_absolute_objective_gap": worst["objective_abs_gap"],
        "max_relative_objective_gap": max(row["objective_relative_gap"] for row in rows),
        "max_regional_advantage": max(row["objective_gap_regional_minus_de"] for row in rows),
        "max_de_advantage": max(-row["objective_gap_regional_minus_de"] for row in rows),
        "support_matches": sum(row["support_match"] for row in rows),
        "support_mismatches": sum(not row["support_match"] for row in rows),
        "strengthened_cases": sum(row["strengthened"] for row in rows),
        "worst_case": {"label": worst["label"], "regime": worst["regime"]},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    tasks = primary_tasks()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            print(
                f"{index:02d}/{len(tasks)} {row['label']} {row['regime']} "
                f"gap={row['objective_gap_regional_minus_de']:.6g}",
                flush=True,
            )
    rows.sort(key=lambda row: (row["label"], row["regime"]))
    payload = {"summary": summarize(rows), "rows": rows}
    (OUT / "regional_factorial_validation.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False)
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
