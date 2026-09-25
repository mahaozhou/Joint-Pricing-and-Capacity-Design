# Joint Pricing and Capacity Design - computational materials

This repository contains the computational materials used for the numerical analyses in **Strategic Patient Choice in Integrated Outpatient Care: Joint Pricing and Capacity Design with Downstream Follow-Up**.

## Data availability

The underlying department-level operational data used for empirical calibration cannot be publicly released due to a confidentiality agreement. The numerical experiments can nevertheless be reproduced using the model primitives, calibrated quantities, and sensitivity ranges reported in the manuscript and Supplemental Online Materials. The files in `results/` are saved computational outputs and do not contain the confidential raw operational data.

## Repository structure

```text
Joint-Pricing-and-Capacity-Design/
├── code/                 # model, optimization, experiments, simulation, reporting
├── tests/                # unit and cross-validation tests
├── results/
│   ├── main/             # principal numerical design and policy-transfer results
│   ├── staffing/         # parallel-clinician / M/M/k results
│   ├── simulation/       # DES summaries and replication-level arrays
│   ├── state/            # state-observing experiments
│   └── tables/           # generated LaTeX tables and text assets
├── figures/              # generated publication figures
├── requirements.txt      # Python package versions
└── README.md
```

## Environment

The reported computations used Python 3.11 on Windows 11. Install the required packages from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Verification

Run the included tests from the repository root:

```bash
python -m pytest tests -q
```

The packaged version passes all 10 tests.

## Main workflow

Run the following commands from the repository root to repeat the numerical searches and then regenerate the exhibits. The saved outputs are already included. To regenerate only the tables and figures from those outputs, run the two reporting commands shown below the full workflow.

```bash
python code/regional_validation.py --workers 4
python code/experiments.py --workers 4
python code/aware_benchmark.py
python code/staffing_referral_benchmark.py
python code/reserve_sensitivity.py
python code/many_server_audit.py
python code/report_many_server_audit.py
python code/simulation.py
python code/additional_validation.py
python code/refresh_statistics.py
python code/report.py
```

To rebuild the current exhibit files without rerunning the optimization or simulation experiments:

```bash
python code/report_many_server_audit.py
python code/report.py
```

`code/refresh_statistics.py` recomputes summary intervals from the saved replication arrays in both `results/simulation/` and `results/state/`. Run it after the simulation commands if those arrays are regenerated. The reporting script reads the saved summaries and does not itself rerun the simulations.

### Code roles

- `code/model.py`: model primitives, equilibrium evaluation, and optimization routines.
- `code/regional_solver.py`: continuous-capacity regional solution.
- `code/regional_validation.py`: independent cross-validation of the regional solution.
- `code/experiments.py`: principal factorial design and policy comparisons.
- `code/aware_benchmark.py` and `code/reserve_sensitivity.py`: policy-transfer and capacity-buffer checks.
- `code/many_server_audit.py` and `code/staffing_referral_benchmark.py`: parallel-clinician staffing analyses.
- `code/simulation.py` and `code/additional_validation.py`: patient-level DES and state-observing checks.
- `code/report.py` and `code/report_many_server_audit.py`: generation of tables and figures from saved outputs. The former produces separate pathway-comparison and payment-arrangement figures.

## Saved outputs

- `results/main/experiments.json`: principal configured policy results.
- `results/main/all_policies.csv`: flat export of the configured policies, including the independent $\delta_E$ sensitivity.
- `results/main/regional_factorial_validation.json`: independent regional-solution validation.
- `results/main/transfers.json` and `results/main/aware_transfers.json`: simplified-model policy-transfer comparisons.
- `results/staffing/`: parallel-clinician staffing, pricing, and transfer results.
- `results/simulation/des_*.json`: simulation configurations and summary statistics.
- `results/simulation/des_*_replications.npz`: replication-level simulation outputs.
- `results/state/`: state-observing screening, final validation, paired comparisons, and replication arrays.
- `results/tables/`: generated LaTeX tables, including `deltaE_table.tex`, `validation_cases.tex`, `delay_details.tex`, and `utility_details.tex`.
- `figures/`: generated figure PDFs and PNG previews. Main-text Figure 4 is `pathway_comparison.pdf`; Supplemental Figure C.1 is `payment_arrangements.pdf`.
