"""Render the appendix figures from the archived numerical results."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from build_section6_figures import HATCHED_BAR, INK, SECONDARY, STYLE, VIRTUAL


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def independent_probabilities(results):
    experiments = pd.read_csv(results / "experiments.csv")
    cases = experiments.loc[experiments.experiment.eq("independent_ab") & experiments.theta.eq(0)]
    fig, axes = plt.subplots(
        2, 3, figsize=(6.7, 4.05), sharex=True, sharey=True, layout="constrained"
    )
    entry_styles = {
        "virtual": ("Virtual-first", "o", VIRTUAL),
        "in-person": ("In-person-first", "s", "white"),
        "both": ("Mixed entry", "D", SECONDARY),
    }
    delta_1 = np.linspace(0.065, 0.8 / 1.8, 160)
    boundary_style = {"color": "#969696", "linewidth": 0.85, "linestyle": "--"}

    for row, demand in enumerate((572, 878)):
        for col, regime in enumerate(("F0", "F1", "F2")):
            ax = axes[row, col]
            subset = cases.loc[cases.demand.eq(demand) & cases.regime.eq(regime)]
            ax.plot(delta_1, delta_1 / (1 - delta_1), **boundary_style)
            ax.set_title(fr"$\Lambda={demand}$, {regime}")
            for entry, (_, marker, face) in entry_styles.items():
                points = subset.loc[subset.entry.eq(entry)]
                ax.plot(
                    points.a, points.b, linestyle="none", marker=marker, markersize=4.9,
                    color=INK, markerfacecolor=face, markeredgewidth=0.7,
                )
            for point in subset.itertuples(index=False):
                ax.annotate(
                    str(point.k_v), (point.a, point.b), xytext=(0, 4.5),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8,
                )
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#e3e3e3", linewidth=0.55)

    axes[0, 0].set(
        xlim=(0.065, 0.84), ylim=(0.03, 0.8),
        xticks=[0.1, 0.2, 0.35, 0.5, 0.65, 0.8],
        xticklabels=[".10", ".20", ".35", ".50", ".65", ".80"],
        yticks=[0.1, 0.35, 0.65], yticklabels=[".10", ".35", ".65"],
    )
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\delta_2$")
    for ax in axes[-1]:
        ax.set_xlabel(r"$\delta_1$")

    handles = [
        Line2D([], [], color=INK, linestyle="none", marker=marker,
               markerfacecolor=face, markersize=4.9, markeredgewidth=0.7, label=label)
        for label, marker, face in entry_styles.values()
    ]
    handles.append(Line2D([], [], **boundary_style, label=r"$t_v=t_f$"))
    fig.legend(handles=handles, loc="outside lower center", ncols=4, handlelength=1.5)
    return fig


def tied_allocations(results):
    evidence = pd.read_csv(results / "exact_design_tie_set.csv")
    ties = evidence.loc[evidence.status.str.startswith("exact")].sort_values("k_v")
    fig, (flow, delay) = plt.subplots(1, 2, figsize=(6.55, 2.85), layout="constrained")
    bars = {"width": 0.67, "linewidth": 0.6}
    virtual = flow.bar(ties.k_v, ties.x_v, color=VIRTUAL, edgecolor=INK, label="Virtual-first", **bars)
    in_person = flow.bar(
        ties.k_v, ties.x_f, bottom=ties.x_v, label="In-person-first",
        **HATCHED_BAR, **bars,
    )
    flow.set(
        title="(a) Initial-channel flows", ylabel="Strategic arrivals per day",
        ylim=(0, 635), yticks=[0, 200, 400, 600],
    )
    fig.legend(
        handles=[virtual, in_person], loc="outside lower center",
        ncols=2, handlelength=1.35, columnspacing=1.2,
    )

    delay.plot(ties.k_v, ties.mandatory_minutes, "o", color=VIRTUAL, markersize=4.7)
    delay.set(
        title="(b) Mandatory-patient sojourns", ylabel="Episode sojourn time (min)",
        ylim=(15.8, 21.35), yticks=[16, 17, 18, 19, 20, 21],
    )
    for ax in (flow, delay):
        ax.set(xlabel=r"Virtual clinicians $k_v$", xlim=(2.45, 13.55), xticks=ties.k_v)
    return fig


def recurrent_policies(results):
    transfers = pd.read_csv(results / "recurrent_transfers.csv")
    sweep = transfers.loc[transfers.comparison.eq("recurrent_switch")].sort_values(["gamma", "b"])
    fig, (staffing, loss) = plt.subplots(
        1, 2, figsize=(6.55, 2.65), sharex=True, layout="constrained"
    )
    styles = {
        0: ("o", 3.0, INK, INK),
        0.5: ("s", 4.6, SECONDARY, "white"),
        1: ("^", 6.6, VIRTUAL, "white"),
    }
    for gamma, group in sweep.groupby("gamma", sort=False):
        marker, size, color, face = styles[gamma]
        style = {
            "linestyle": "none", "marker": marker, "markersize": size, "color": color,
            "markerfacecolor": face, "markeredgewidth": 0.8, "zorder": 4 - gamma * 2,
        }
        staffing.plot(group.b, group.k_v, label=fr"$\gamma={gamma:g}$", **style)
        # Near-zero values differ only by floating-point roundoff.
        values = group.loss_candidate.mask(group.loss_candidate.abs().lt(1e-8), 0)
        loss.plot(group.b, values, **style)
    for ax in (staffing, loss):
        ax.axvline(0.25, color="#969696", linewidth=0.85, linestyle="--", zorder=1)
        ax.set(xlabel=r"$\delta_2$", xlim=(0.194, 0.356), xticks=[0.20, 0.25, 0.30, 0.35])
    staffing.set(
        title="(a) Selected clinician allocations", ylabel=r"Virtual clinicians $k_v$",
        ylim=(0.8, 15.3), yticks=[3, 6, 9, 12, 15],
    )
    staffing.legend(loc="upper right", handlelength=1.1, labelspacing=0.32)
    loss.set(
        title="(b) Staged-policy implementation loss", ylabel="Objective loss",
        ylim=(-150, 3550), yticks=[0, 1000, 2000, 3000],
    )
    loss.axhline(0, color="#c7c7c7", linewidth=0.65, zorder=1)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figs")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        for name, plot in (
            ("figC1_independent", independent_probabilities),
            ("figC2_ties", tied_allocations),
            ("figE1_recurrence", recurrent_policies),
        ):
            fig = plot(args.results_dir)
            fig.savefig(args.output_dir / f"{name}.pdf")
            plt.close(fig)


if __name__ == "__main__":
    main()
