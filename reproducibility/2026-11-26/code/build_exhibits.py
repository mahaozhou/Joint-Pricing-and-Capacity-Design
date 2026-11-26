"""Build the current manuscript's tables and plotting inputs from experiment outputs."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import pandas as pd

from des import summarize


def loss_summary(rows, keys, estimate="loss_candidate", threshold=0.0):
    return (
        rows.assign(certified=rows["loss_lower"] > threshold)
        .groupby(keys, sort=True)
        .agg(
            cases=(estimate, "size"),
            positive=("certified", "sum"),
            median_loss=(estimate, "median"),
            max_loss=(estimate, "max"),
            q1=(estimate, lambda values: values.quantile(0.25)),
            q3=(estimate, lambda values: values.quantile(0.75)),
        )
        .reset_index()
    )


def matched_effect(before, after, keys):
    paired = before.merge(after, on=keys, suffixes=("_before", "_after"), validate="one_to_one")
    return {
        "cases": len(paired),
        "certified_objective_improvements": int(
            (paired.lower_bound_after > paired.upper_bound_before).sum()
        ),
        "median_delta_J": (paired.J_after - paired.J_before).median(),
        "allocation_changes": int((paired.k_v_before != paired.k_v_after).sum()),
        "entry_changes": int((paired.entry_before != paired.entry_after).sum()),
        "median_delta_access_pp": 100 * (paired.access_after - paired.access_before).median(),
        "median_delta_D_E_min": (paired.mandatory_minutes_after - paired.mandatory_minutes_before).median(),
    }


def sensitivity(experiments, levels):
    frame = experiments[experiments.experiment == "single_factor"]
    rows = []
    for factor, reference in levels.items():
        cases = frame[frame.varied == factor]
        baseline = cases[cases[factor] == reference]
        for level, group in cases.groupby(factor):
            rows.append({
                "factor": factor,
                "level": level,
                **matched_effect(baseline, group, ["demand", "b", "regime"]),
            })
    return pd.DataFrame(rows)


def plot_inputs(experiments, paths, results, directory):
    test = experiments.query(
        "experiment == 'paths' and demand == 878 and s == 30 and b == 0.4 and theta == 0 and regime == 'F1'"
    ).set_index("path")
    origin, target = test.loc["O"], test.loc["S"]
    transfer = paths[
        (paths.comparison == "O->S") & (paths.source_id == origin.scenario_id)
        & (paths.destination_id == target.scenario_id)
    ].iloc[0]
    rows = [
        dict(policy="O optimized in O", k_v=origin.k_v, k_f=origin.k_f, x_v=origin.x_v,
             x_f=origin.x_f, J=origin.J, p_v=origin.p_v, access=origin.access,
             mandatory_minutes=origin.mandatory_minutes),
        dict(policy="O policy in S", k_v=origin.k_v, k_f=origin.k_f, x_v=transfer.transfer_x_v,
             x_f=transfer.transfer_x_f, J=transfer.transfer_J, p_v=origin.p_v, access=transfer.transfer_access,
             mandatory_minutes=transfer.transfer_mandatory_minutes),
        dict(policy="S optimized in S", k_v=target.k_v, k_f=target.k_f, x_v=target.x_v,
             x_f=target.x_f, J=target.J, p_v=target.p_v, access=target.access,
             mandatory_minutes=target.mandatory_minutes),
    ]
    pd.DataFrame(rows).to_csv(directory / "figure2_case.csv", index=False)

    curves = experiments[
        experiments.experiment.isin(["followup_scan", "followup_refined"])
        & (experiments.s == 15) & (experiments.theta == 0.5) & (experiments.regime == "F1")
    ].sort_values(["demand", "b"]).drop_duplicates(["demand", "b"])
    curves.assign(followup_load=curves.ell_v - curves.x_v).to_csv(
        directory / "figure3_reference_curves.csv", index=False
    )

    witnesses = json.loads((results / "exact_checks.json").read_text(encoding="utf-8"))["scale_witness"]
    rows = []
    for label, delta2, point in (
        ("Below", 0.245, witnesses["left"][0]),
        ("Threshold: virtual", 0.25, witnesses["threshold_virtual"]),
        ("Threshold: in-person", 0.25, witnesses["threshold_in_person"]),
        ("Above", 0.255, witnesses["right"][0]),
    ):
        x_v, x_f = (value["decimal"] for value in point["x"])
        ell_v, ell_f = (value["decimal"] for value in point["load"])
        w_v, w_f = (value["decimal"] for value in point["W"])
        rows.append(dict(policy=label, delta2=delta2, k_v=point["k"][0], k_f=point["k"][1],
                         x_v=x_v, x_f=x_f, ell_v=ell_v, ell_f=ell_f, W_v=w_v, W_f=w_f,
                         J=point["J"]["decimal"], p_v=point["p"][0]["decimal"], followup_load=ell_v-x_v))
    pd.DataFrame(rows).to_csv(directory / "figure4_exact_witnesses.csv", index=False)
    tied = pd.read_csv(results / "exact_design_tie_set.csv")
    tied[tied.status.str.startswith("exact")].to_csv(directory / "figure4_threshold_optimal_staffing.csv", index=False)
    effects = pd.read_csv(results / "patient_impacts.csv")
    effects[effects.kind != "reference"].to_csv(directory / "figure5_patient_comparisons.csv", index=False)


def des_tables(results, tables):
    summaries, by_stage = [], []
    for prefix in ("des", "des_extended"):
        replications = pd.read_csv(results / f"{prefix}_replications.csv")
        source = pd.read_csv(results / f"{prefix}_summary.csv")[["case", "metric", "analytic"]]
        measures = ["W_v", "W_f", "D_E", "J", "U_v_probe", "U_f_probe", "U_difference_probe"]
        means = summarize(replications, ["case"], measures)
        means = means.merge(source, on=["case", "metric"], validate="one_to_one")
        means["inside_95"] = means.analytic.between(means.lower_95, means.upper_95)
        summaries.append(means.assign(run=prefix))

        raw = pd.read_csv(results / f"{prefix}_stages_raw.csv")
        expected = pd.read_csv(results / f"{prefix}_stage_summary.csv")
        expected = expected[expected.metric == "W"][["case", "route", "stage", "metric", "pool_analytic"]]
        intervals = summarize(raw, ["case", "route", "stage"], ["W"]).merge(
            expected, on=["case", "route", "stage", "metric"], validate="one_to_one"
        )
        intervals["inside_95"] = intervals.pool_analytic.between(intervals.lower_95, intervals.upper_95)
        by_stage.append(intervals.assign(run=prefix))

    combined = pd.concat(summaries, ignore_index=True)
    for table, metrics in (("F1", ["W_v", "W_f"]), ("F2", ["D_E", "J"]), ("F_utilities", measures[4:])):
        subset = combined[combined.metric.isin(metrics)].copy()
        subset["unit"] = np.where(subset.metric.isin(["W_v", "W_f", "D_E"]), "minutes", "normalized units")
        for name in ("mean", "lower_95", "upper_95", "analytic"):
            subset[name] = np.where(subset.unit == "minutes", 480 * subset[name], subset[name])
        subset.to_csv(tables / f"table_{table}.csv", index=False)
    stages = pd.concat(by_stage, ignore_index=True)
    for name in ("mean", "lower_95", "upper_95", "pool_analytic"):
        stages[name] *= 480
    stages.to_csv(tables / "table_F_stages_minutes.csv", index=False)


def build(results, output):
    tables = output / "tables"
    plots = output / "plot_data"
    tables.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    experiments = pd.read_csv(results / "experiments.csv")
    reference = experiments[(experiments.experiment == "paths") & (experiments.path == "S")]
    inputs = ["demand", "K", "nu_v", "nu_f", "required", "V", "T", "s", "cv", "cf", "penalty",
              "a", "d", "b", "theta", "alpha", "regime", "price_lo_v", "price_hi_v", "price_lo_f", "price_hi_f"]
    reference[inputs].sort_values(["regime", "demand", "s", "b", "theta"]).to_csv(
        tables / "table_1_settings.csv", index=False
    )

    paths = pd.read_csv(results / "path_transfers.csv")
    summary = loss_summary(paths, ["comparison", "regime"])
    details = paths.groupby(["comparison", "regime"]).agg(
        entry_changes=("entry_changed", "sum"),
        allocation_changes=("staff_changed", "sum"),
        access_loss_pp=("delta_access", lambda values: 100 * values.median()),
    ).reset_index()
    summary.merge(details, on=["comparison", "regime"]).to_csv(tables / "table_2.csv", index=False)

    repricing = pd.read_csv(results / "repricing.csv")
    repricing = repricing[
        (repricing.experiment == "followup_scan") & ~np.isclose(repricing.b, repricing.reference_b_fixed)
    ]
    for label, group in (("3", ["regime", "theta"]), ("C4", ["demand", "regime"])):
        summary = loss_summary(repricing, group, estimate="candidate_loss", threshold=0.1)
        within = repricing.assign(within=repricing.loss_upper <= 0.1).groupby(group).within.sum().reset_index()
        summary.merge(within, on=group).to_csv(tables / f"table_{label}.csv", index=False)

    examples = experiments[
        experiments.experiment.isin(["exact_small_role", "exact_small_gain"]) & (experiments.path == "S")
    ].copy()
    examples["s_minus_T"] = examples.s - examples["T"]
    examples[["experiment", "K", "demand", "required", "a", "b", "d", "V", "s_minus_T",
              "cv", "cf", "alpha", "penalty", "theta", "cap_v", "cap_f"]].to_csv(
        tables / "table_B1.csv", index=False
    )

    loss_summary(paths, ["regime", "theta", "comparison"]).to_csv(tables / "table_C1.csv", index=False)
    grid = experiments[experiments.experiment == "independent_ab"].copy()
    grid["efficient_virtual_exclusive_in_person"] = (
        (grid.clinician_time_difference_minutes < 0) & (grid.entry == "in-person")
    )
    counts = grid.groupby(["regime", "theta", "entry"]).size().unstack(fill_value=0).reset_index()
    efficiency = grid.groupby(["regime", "theta"]).efficient_virtual_exclusive_in_person.sum().reset_index()
    counts.merge(efficiency, on=["regime", "theta"]).to_csv(tables / "table_C2.csv", index=False)

    comparisons = []
    for before, after in (("F0", "F1"), ("F1", "F2"), ("F0", "F2")):
        comparisons.append(dict(comparison=f"{before}->{after}", **matched_effect(
            reference[reference.regime == before], reference[reference.regime == after],
            ["demand", "s", "b", "theta"],
        )))
    pd.DataFrame(comparisons).to_csv(tables / "table_C3.csv", index=False)
    sensitivity(experiments, {"alpha": 1.0, "nu_v": 54.0}).to_csv(tables / "table_C5.csv", index=False)
    sensitivity(experiments, {"required": 30.0, "d": 0.25}).to_csv(tables / "table_D2.csv", index=False)

    accounts = pd.read_csv(results / "mandatory_accounts_transfers.csv")
    combined = pd.concat([accounts, accounts.assign(regime="All")], ignore_index=True)
    loss_summary(combined, ["comparison", "regime"]).to_csv(tables / "table_D1.csv", index=False)
    attendance = pd.read_csv(results / "elasticity_transfers.csv")
    range_ = attendance.groupby("elasticity").E.agg(E_min="min", E_max="max").reset_index()
    loss_summary(attendance, ["elasticity"]).merge(range_, on="elasticity").to_csv(tables / "table_D3.csv", index=False)

    recurrence = pd.read_csv(results / "recurrent_transfers.csv").query("comparison == 'recurrent'")
    allocation = recurrence.groupby("gamma").staff_changed.sum().rename("allocation_changes").reset_index()
    loss_summary(recurrence, ["gamma"]).merge(allocation, on="gamma").to_csv(tables / "table_E1.csv", index=False)

    des_tables(results, tables)
    columns = ["kv", "baseline_virtual_share", "observed_virtual_share", "baseline_DE", "observed_DE",
               "delta_DE_min", "delta_DE_min_lo", "delta_DE_min_hi"]
    pd.read_csv(results / "state_observing" / "paired_summary.csv")[columns].to_csv(
        tables / "table_G1.csv", index=False
    )
    plot_inputs(experiments, paths, results, plots)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.results_dir, args.output_dir)
