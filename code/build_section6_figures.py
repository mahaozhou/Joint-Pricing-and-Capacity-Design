"""Render the Section 6 figures from the supplied result tables."""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "exhibits" / "plot_data"
INK = "#242424"
VIRTUAL = "#344f63"
SECONDARY = "#5b7486"
HATCHED_BAR = {"color": "#e1e8ec", "edgecolor": "#5c646a", "hatch": "///"}
SOLID_LINE = {"color": VIRTUAL, "marker": "o", "linestyle": "-"}
DASHED_LINE = {
    "color": SECONDARY,
    "marker": "s",
    "markerfacecolor": "white",
    "linestyle": "--",
}

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "axes.linewidth": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "legend.borderaxespad": 0.4,
    "hatch.linewidth": 0.4,
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
}


def pathways(data):
    cases = pd.read_csv(data / "figure2_case.csv").set_index("policy").loc[
        ["O optimized in O", "O policy in S", "S optimized in S"]
    ]
    names = [r"$O$ in $O$", r"$O$ in $S$", r"$S$ in $S$"]
    fig, axes = plt.subplots(1, 3, figsize=(6.55, 2.36), layout="constrained")
    staffing, arrivals, objective = axes
    positions = range(len(cases))
    bar = {"width": 0.64, "linewidth": 0.65}

    virtual = staffing.bar(positions, cases.k_v, color=VIRTUAL, edgecolor=INK, **bar)
    in_person = staffing.bar(positions, cases.k_f, bottom=cases.k_v, **HATCHED_BAR, **bar)
    staffing.bar_label(virtual, label_type="center", color="white")
    staffing.bar_label(in_person, label_type="center", color=INK)
    staffing.set(
        title="(a) Clinician allocation", ylabel="Clinicians", ylim=(0, 23),
        yticks=[0, 6, 12, 18], xticks=positions, xticklabels=names,
    )
    staffing.legend([virtual, in_person], ["Virtual", "In-person"],
                    loc="upper center", ncols=2, handlelength=1.2, columnspacing=0.9)

    arrivals.bar(positions, cases.x_v, color=VIRTUAL, edgecolor=INK, **bar)
    totals = arrivals.bar(positions, cases.x_f, bottom=cases.x_v, **HATCHED_BAR, **bar)
    arrivals.bar_label(totals, [f"{rate:.1%}" for rate in cases.access], padding=3)
    arrivals.set(
        title="(b) Initial channel", ylabel="Strategic arrivals per day", ylim=(0, 790),
        yticks=[0, 200, 400, 600], xticks=positions, xticklabels=names,
    )

    in_complete_model = cases.J.iloc[1:]
    bars = objective.bar([0, 1], in_complete_model / 1_000, color="#a4a4a4", edgecolor=INK, **bar)
    objective.bar_label(bars, fmt="%.2f", padding=3)
    loss = 1 - in_complete_model.iloc[0] / in_complete_model.iloc[1]
    objective.text(0.5, 28.2, f"{loss:.2%} loss", ha="center", va="bottom")
    objective.set(
        title=r"(c) Objective in $S$", ylabel=r"$J$ (thousands)", ylim=(0, 31),
        yticks=[0, 10, 20, 30], xticks=[0, 1], xticklabels=names[1:],
    )
    return fig


def reference_scans(data):
    scans = pd.read_csv(data / "figure3_reference_curves.csv").sort_values(["demand", "b"])
    fig, axes = plt.subplots(
        4, 2, figsize=(6.55, 5.25), sharex=True, sharey="row", layout="constrained"
    )
    for col, ((demand, scan), panel) in enumerate(zip(scans.groupby("demand"), ["a", "b"])):
        staffing, arrivals, load, price = axes[:, col]
        staffing.set_title(fr"({panel}) $\Lambda={demand:g}$")
        staffing.plot(scan.b, scan.k_v, color=VIRTUAL, marker="o", linestyle="none", markersize=2.7)

        # Break visual guides at changes in the selected allocation or entry pattern.
        policy = scan[["k_v", "entry"]]
        runs = policy.ne(policy.shift()).any(axis=1).cumsum()
        for run_number, (_, run) in enumerate(scan.groupby(runs)):
            for axis, field, style, label in (
                (arrivals, "x_v", SOLID_LINE, "Virtual-first"),
                (arrivals, "x_f", DASHED_LINE, "In-person-first"),
                (load, "ell_v", SOLID_LINE, "Total"),
                (load, "followup_load", DASHED_LINE, "Follow-up"),
                (price, "p_v", SOLID_LINE, None),
            ):
                axis.plot(
                    run.b, run[field], **style, linewidth=0.9, markersize=2.4,
                    markeredgewidth=0.65, label=label if run_number == 0 else None,
                )

    axes[0, 0].set(ylabel=r"Virtual clinicians $k_v$", ylim=(0, 16), yticks=[0, 4, 8, 12, 16])
    axes[1, 0].set(ylabel="Strategic arrivals", ylim=(-40, 900), yticks=[0, 400, 800])
    axes[2, 0].set(ylabel="Virtual encounters", ylim=(-35, 735), yticks=[0, 300, 600])
    axes[3, 0].set(ylabel=r"Virtual price $p_v$", ylim=(17, 32), yticks=[20, 25, 30])
    axes[1, 1].legend(loc="upper right")
    axes[2, 1].legend(loc="center right")
    for ax in axes[-1]:
        ax.set(xlabel=r"$\delta_2$", xlim=(-0.016, 0.816), xticks=[0, 0.2, 0.4, 0.6, 0.8])
    return fig


def designed_threshold(data):
    witnesses = pd.read_csv(data / "figure4_exact_witnesses.csv")
    counts = pd.read_csv(data / "figure4_threshold_optimal_staffing.csv")
    endpoints = witnesses.iloc[[0, -1]]
    threshold = witnesses.delta2.iloc[1]
    fig, (staffing, load) = plt.subplots(1, 2, figsize=(6.55, 2.7), layout="constrained")

    staffing.axvline(threshold, color="#aeaeae", linestyle="--", linewidth=0.75, zorder=1)
    staffing.plot(
        endpoints.delta2, endpoints.k_v, "o", color=VIRTUAL, markersize=5,
        linestyle="none", label="Unique at endpoint",
    )
    staffing.plot(
        [threshold] * len(counts), counts.k_v, "o", color=INK, markersize=4.4,
        markerfacecolor="white", markeredgewidth=0.85, linestyle="none", label="Tied at threshold",
    )
    staffing.set(
        title="(a) Optimal allocation", xlabel=r"$\delta_2$", ylabel=r"Virtual clinicians $k_v$",
        xlim=(0.2438, 0.2562), ylim=(0.4, 17.6),
        xticks=[endpoints.delta2.iloc[0], threshold, endpoints.delta2.iloc[1]],
        xticklabels=[f"{x:.3f}" for x in [endpoints.delta2.iloc[0], threshold, endpoints.delta2.iloc[1]]],
        yticks=[1, 3, 5, 7, 9, 11, 13, 15, 17],
    )
    staffing.legend(loc="lower left", handlelength=1.1)

    positions = range(len(witnesses))
    bar = {"width": 0.65, "linewidth": 0.65}
    load.bar(positions, witnesses.x_v, color=VIRTUAL, edgecolor=INK, label="Initial", **bar)
    followup = load.bar(
        positions, witnesses.followup_load, bottom=witnesses.x_v,
        label="Follow-up", **HATCHED_BAR, **bar,
    )
    load.bar_label(followup, [f"{value:.2f}" for value in witnesses.ell_v], padding=3)
    load.set(
        title="(b) Virtual workload", ylabel="Virtual encounters per day",
        ylim=(0, 815), yticks=[0, 200, 400, 600, 800], xticks=positions,
        xticklabels=["0.245", "0.250\nVirtual-first", "0.250\nIn-person-first", "0.255"],
    )
    load.legend(loc="upper center", ncols=2, handlelength=1.3, columnspacing=1.0)
    return fig


def patient_sojourn(data):
    cases = pd.read_csv(data / "figure5_patient_comparisons.csv").set_index("kind")
    fig, ax = plt.subplots(figsize=(5.35, 2.7), layout="constrained")
    ax.axhline(0, color="#989898", linewidth=0.7)
    for case, style, label, offset, align in (
        ("exact-cross-point", SOLID_LINE, "Across threshold", (-4, 6), "right"),
        ("exact-common-threshold", DASHED_LINE, "At threshold", (5, -7), "left"),
    ):
        row = cases.loc[case]
        ax.plot(
            [0, 1], [row.delta_d0, row.delta_d1], color=style["color"],
            linestyle=style["linestyle"], linewidth=1.4, label=label,
        )
        ax.plot(row.d_critical, 0, **style, markersize=4.8, markeredgewidth=0.9)
        ax.annotate(
            f"{row.d_critical:.4f}", (row.d_critical, 0), xytext=offset, textcoords="offset points",
            ha=align, va="bottom" if offset[1] > 0 else "top", fontsize=9,
        )
    ax.set(
        xlabel=r"Subgroup follow-up probability $\delta_{E,i}$",
        ylabel="Episode sojourn time change (min)",
        xlim=(0, 1), ylim=(-10, 42), yticks=[-10, 0, 10, 20, 30, 40],
    )
    ax.legend(loc="upper left")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figs")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        for name, plot in (
            ("fig2_pathways", pathways),
            ("fig3_reference", reference_scans),
            ("fig4_threshold", designed_threshold),
            ("fig5_patients", patient_sojourn),
        ):
            fig = plot(args.data_dir)
            fig.savefig(args.output_dir / f"{name}.pdf")
            plt.close(fig)


if __name__ == "__main__":
    main()
