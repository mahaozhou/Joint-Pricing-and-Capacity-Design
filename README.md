# Joint Pricing and Capacity Design - computational materials

This repository contains the computational materials used for the numerical analyses in **Strategic Patient Choice in Integrated Online and Offline Outpatient Services: Joint Pricing and Capacity Design**.

## Data availability

The underlying department-level operational data used for empirical calibration are not included because they are subject to a confidentiality agreement. The numerical experiments can be reproduced from the model primitives, calibrated quantities, and sensitivity ranges reported in the manuscript and Supplemental Online Materials. The files under `results/revision/` are saved computational outputs, not the confidential raw operational data.

## Environment

- Python 3.11
- Windows 11 was used for the reported runs
- Required Python packages are listed in `revision/requirements.txt`

Install dependencies from the repository root with:

```bash
python -m pip install -r revision/requirements.txt
```

## Main computational workflow

Run from the repository root:

```bash
python -m pytest revision/test_model.py -q
python -m pytest revision/test_regional_solver.py -q
python revision/regional_validation.py --workers 4
python revision/experiments.py --workers 4
python revision/aware_benchmark.py
python revision/staffing_referral_benchmark.py
python revision/reserve_sensitivity.py
python revision/many_server_audit.py
python revision/report_many_server_audit.py
python revision/simulation.py
python revision/additional_validation.py
python revision/refresh_statistics.py
python revision/report.py
```

`revision/model.py` contains the common analytical and equilibrium routines. `revision/regional_solver.py` and `revision/regional_validation.py` implement the continuous-capacity regional solution and independent validation. `revision/experiments.py` generates the principal numerical design. The staffing, simulation, and state-observing checks are implemented in the correspondingly named scripts.

## Saved outputs

- `results/revision/experiments.json`: principal configured policy results
- `results/revision/regional_factorial_validation.json`: independent regional-solution validation
- `results/revision/transfers.json` and `aware_transfers.json`: policy-transfer comparisons
- `results/revision/staffing_transfers.json` and `staffing_referral_transfers.json`: parallel-clinician comparisons
- `results/revision/many_server_audit/`: selected multi-server audit outputs
- `results/revision/des_*.json`: simulation configurations and summary statistics
- `results/revision/des_*_replications.npz`: replication-level simulation outputs
- `results/revision/state_screening.json`, `state_validation.json`, and `state_paired.json`: state-observing analyses
- `results/revision/*.tex` and `figs/revision/`: generated reporting artifacts and publication-snapshot figures

The three Python scripts at the repository root retain the historical filenames referenced in the earlier reproducibility workflow. The revised computational pipeline used for the current manuscript is under `revision/`.
