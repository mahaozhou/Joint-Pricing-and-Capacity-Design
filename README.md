# IISE Transactions replication package

**Strategic Patient Choice in Integrated Outpatient Care: Joint Pricing and Clinician Allocation with Downstream Follow-Up**

This package corresponds to the integer-clinician manuscript and Supplemental Material A–G prepared for the November 26, 2026 resubmission. It includes the complete care pathway, certified price and staffing optimization, mandatory-demand comparisons, recurrent routing, stationary simulation, and the congestion-information experiment.

[Reproducibility report (PDF)](Reproducibility_Report.pdf)

## Data and units

The underlying departmental operating records are covered by a confidentiality agreement. They inform the strategic demand scales of 572 and 878 and the clinician team size of 18. The remaining numerical settings are specified by the model and experimental design. The released inputs and outputs consist of model parameters, numerical searches, and simulated replications. The code reads these released files and generates arrivals from the specified stochastic models. Time is measured in eight-hour clinic-day units and is converted to minutes in the sojourn-time exhibits. Monetary and utility quantities use the manuscript's normalized units.

## Software and execution

The original environment is recorded in `results/environment.txt`. Use Python 3.11 or 3.12 and install the pinned direct dependencies:

```sh
python -m pip install -r requirements.txt
```

Run these commands from this directory. Each command writes to the indicated output directory.

| Purpose | Command |
|---|---|
| Rebuild the current tables and plots from saved outputs | `python code/reproduce.py --mode exhibits --output-dir reproduced-exhibits` |
| Repeat the prescribed-flow and state-observing simulations, then rebuild the exhibits | `python code/reproduce.py --mode simulations --workers 4 --output-dir reproduced-simulations` |
| Start with the archived search ledger, recompute policy transfers and independent checks, rerun both simulations, and rebuild exhibits | `python code/reproduce.py --mode all --reuse-ledger --workers 4 --output-dir reproduced-ledger` |
| Start the price and staffing searches from scratch and repeat the downstream analyses and simulations | `python code/reproduce.py --mode all --workers 4 --output-dir reproduced-full` |

Optimization and simulation require GNU C++17 with OpenMP. The verified compiler version is 14.2.0. The runner uses `g++` on the system path or the executable specified by `CXX`. The kernel is compiled with floating-point contraction disabled for the outward objective bounds. Use an empty output directory for a fresh search. Repeating a full command in the same directory resumes the search ledger. The exhibit-only calculation uses the Python dependencies. The schematic is rebuilt when pdfLaTeX and TikZ with the `standalone` class are installed. The supplied figure is used when pdfLaTeX is unavailable.

## Exhibit index

The submitted vector files are in `figures/`. The rebuilt files appear under the chosen output directory. Tabular outputs are CSVs with the indicated table identifiers under `exhibits/tables/`. Figure inputs are under `exhibits/plot_data/` and in the linked result files.

| Manuscript result | Numerical inputs and outputs | Implementation |
|---|---|---|
| Figure 1 and Table 1 | `figure_sources/fig1_pathways.tex`, `table_1_settings.csv` | The TikZ schematic and the scenario registry in `code/experiments.py` |
| Section 4 and Supplemental Material A–B | `raw_searches.jsonl`, `experiments.csv`, `staff_bounds.csv`, `solver_summary.csv`, `independent_audit.json`, `exact_checks.json`, `table_B1.csv` | `model.py`, `kernel.cpp`, `experiments.py`, `verify_selected.py`, `analyze.py`, and `audit_numeric.py`. Table A.1 gives the analytical support conditions. |
| Figure 2, Table 2, Figure C.1, and Tables C.1–C.3 | `experiments.csv`, `path_transfers.csv`, and `figure2_case.csv` | Matched independently optimized models, policy transfers, the independent referral grid, and nested pricing comparisons |
| Figures 3–5, Table 3, Figure C.2, and Table C.4 | `repricing.csv`, `exact_checks.json`, `exact_design_tie_set.csv`, `tie_transfer_bounds.csv`, `patient_impacts.csv` | Fixed-allocation repricing, exact tariff crossings, equivalent policies, and subgroup episode delays |
| Table C.5 and Tables D.1–D.3 | `experiments.csv`, `mandatory_accounts_transfers.csv`, `elasticity_transfers.csv` | One-factor sensitivity, all 108 workload/revenue transfers, and 48 price-responsive attendance cases |
| Figure E.1 and Table E.1 | `recurrent_transfers.csv` | Repeated cross-channel routing, policies and transfers |
| Tables F.1–F.2 and accompanying interval checks | `des_*` and `des_extended_*` result files | `des.py` and `simulate.cpp` for the events, `build_exhibits.py` for the Student t intervals and displayed tables |
| Table G.1 | `results/state_observing/design.json`, `replications.csv`, `extended_k3.csv`, `paired_summary.csv`, `table_G1_rows.tex` | `simulate_state_observing.cpp` and `analyze_state_observing.py` |

The generator `code/build_exhibits.py` assembles the tables and plotting inputs. `code/build_section6_figures.py` builds Figures 2–5 and `code/build_supplement_figures.py` builds Figures C.1, C.2, and E.1. The full-accuracy loss tallies in Table 2 use certified lower bounds strictly above zero. The repricing tallies in Tables 3 and C.4 use lower bounds above 0.1 and upper bounds at most 0.1.

## Simulation and numerical tolerances

- Supplemental Material F uses seven policies with 100 seeded replications per policy. The standard horizon is 50 warm-up and 500 observation units. An additional 100 replications at the high-utilization policy use 100 warm-up and 1,000 observation units. Seeds and matched policy identifiers are included in the replication CSVs.
- Supplemental Material G uses eleven tied policies with 100 paired replications per policy and two choice rules. The common random number specification and extended run for three virtual clinicians are recorded in `results/state_observing/design.json`.
- The reported search ledger contains certified objective-value gaps at most 0.1 for the retained numerical scenarios. High-precision equilibrium checks and exact rational examples are supplied with the code. The precision of the reported numerical tables is the comparison standard across hardware and compiler builds. Alternative optimal or numerically tied policies can have different decision vectors.

## Verification

On Windows 11 with Python 3.11.4, MinGW-w64 GCC 14.2.0, an Intel Core i5-12450H, and four workers, the exhibit-only rebuild took about 13 seconds. A run starting with the archived search ledger rebuilt the analysis, repeated both simulation experiments, and generated the exhibits in about 169 seconds. Both reproduced all 16 numerical tables to the precision reported in the manuscript and rebuilt the eight figures.

A separate run from an empty ledger completed 3,732 distinct optimization searches and the downstream transfer analysis in about 142 seconds. All 4,257 retained design rows achieved objective-value gaps at most 0.1. Its tables were checked against the manuscript using the rerun seeded simulation outputs. Verification timings are recorded in `results/verification.json`.

`SHA256SUMS` records the released files. The generated data are also indexed by exhibit in the table above.
