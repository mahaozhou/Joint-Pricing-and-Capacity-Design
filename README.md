# IISE Transactions replication package

**Strategic Patient Choice in Integrated Outpatient Care: Joint Pricing and Clinician Allocation with Downstream Follow-Up**

[Reproducibility Report (PDF)](Reproducibility_Report.pdf)

The hospital's operating records are confidential. The package includes the model inputs and saved numerical and simulation results.

## Reproduce the results

Run from the repository root with Python 3.11 or 3.12. Optimization and simulation also require `g++` with C++17 and OpenMP support.

```sh
python -m pip install -r requirements.txt
```

| Task | Command |
| --- | --- |
| Rebuild tables and figures from saved results | `python code/reproduce.py --mode exhibits --output-dir reproduced-exhibits` |
| Rerun both simulations and rebuild tables and figures | `python code/reproduce.py --mode simulations --output-dir reproduced-simulations` |
| Rerun the full workflow | `python code/reproduce.py --mode all --output-dir reproduced-full` |

Use an empty output directory for a fresh run. Add `--reuse-ledger` to the full command to start from the archived searches.

Saved results are in `results/`, figures in `figures/`, and tables and figure data in `exhibits/`. The report maps these files to the manuscript and supplement.
