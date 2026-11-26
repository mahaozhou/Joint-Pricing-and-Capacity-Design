"""Rebuild exhibits, repeat both simulations, or repeat the complete computational workflow."""
from argparse import ArgumentParser
from pathlib import Path
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
COMPUTE_FILES = (
    "model.py", "kernel.cpp", "experiments.py", "analyze.py", "audit_numeric.py",
    "verify_selected.py", "des.py", "simulate.cpp",
)


def python_script(script, *options):
    subprocess.run([sys.executable, "-u", str(script), *map(str, options)], check=True)


def run_optimization(code, results, workers, use_ledger):
    if use_ledger and not (results / "raw_searches.jsonl").is_file():
        shutil.copyfile(ROOT / "results" / "raw_searches.jsonl", results / "raw_searches.jsonl")
    python_script(code / "experiments.py", "--workers", workers, "--budget", 9500,
                  "--retry", 45000, "--tight-scenarios")
    python_script(code / "experiments.py", "--no-build", "--workers", workers,
                  "--budget", 9500, "--retry", 500000, "--tight-scenarios")
    python_script(code / "analyze.py")
    python_script(code / "audit_numeric.py")


def run_simulations(code, results, output, workers):
    python_script(code / "des.py", "--workers", workers)
    python_script(code / "des.py", "--workers", workers,
                  "--warmup", 100, "--measurement", 1000, "--high-load-only")

    state = results / "state_observing"
    state.mkdir(exist_ok=True)
    shutil.copyfile(results / "exact_design_tie_set.csv", state / "source_exact_design_tie_set.csv")
    shutil.copyfile(ROOT / "results" / "state_observing" / "design.json", state / "design.json")
    binary = output / ("simulate_state_observing.exe" if os.name == "nt" else "simulate_state_observing")
    compiler = os.environ.get("CXX", "g++")
    platform_flags = ["-static"] if os.name == "nt" else []
    subprocess.run(
        [compiler, "-std=c++17", "-O3", "-fopenmp", *platform_flags,
         str(CODE / "simulate_state_observing.cpp"), "-o", str(binary)],
        check=True,
    )
    policies = state / "source_exact_design_tie_set.csv"
    env = {**os.environ, "OMP_NUM_THREADS": str(workers)}
    for name, target, horizons in (
        ("run", "replications.csv", ["50", "500"]),
        ("extended", "extended_k3.csv", ["100", "1000", "3"]),
    ):
        with (state / f"{name}.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [str(binary), str(policies), str(state / target), "100", *horizons],
                stdout=log, stderr=subprocess.STDOUT, env=env, check=True,
            )


def regenerate_exhibits(results, output):
    python_script(CODE / "analyze_state_observing.py", "--data-dir", results / "state_observing")
    python_script(CODE / "build_exhibits.py", "--results-dir", results, "--output-dir", output / "exhibits")
    images = output / "figures"
    python_script(CODE / "build_section6_figures.py", "--data-dir",
                  output / "exhibits" / "plot_data", "--output-dir", images)
    python_script(CODE / "build_supplement_figures.py", "--results-dir", results, "--output-dir", images)
    if pdflatex := shutil.which("pdflatex"):
        diagram = ROOT / "figure_sources" / "fig1_pathways.tex"
        subprocess.run(
            [pdflatex, "-interaction=batchmode", "-halt-on-error",
             f"-output-directory={images}", str(diagram)],
            check=True,
        )
    else:
        shutil.copyfile(ROOT / "figures" / "fig1_pathways.pdf", images / "fig1_pathways.pdf")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("exhibits", "simulations", "all"), default="exhibits")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduced")
    parser.add_argument("--reuse-ledger", action="store_true", help="start the all mode from the saved search ledger")
    args = parser.parse_args()
    if args.workers < 1 or (args.reuse_ledger and args.mode != "all"):
        parser.error("Use a positive worker count and use --reuse-ledger only with --mode all.")
    output = args.output_dir.resolve()
    protected = (ROOT / "code", ROOT / "results", ROOT / "exhibits", ROOT / "figures")
    if output == ROOT or any(output.is_relative_to(path) for path in protected):
        parser.error("Select an output directory separate from the released files.")
    output.mkdir(parents=True, exist_ok=True)
    results = output / "results"
    if args.mode in ("exhibits", "simulations"):
        shutil.copytree(ROOT / "results", results, dirs_exist_ok=True)
    else:
        results.mkdir(exist_ok=True)

    if args.mode != "exhibits":
        work_code = output / "code"
        work_code.mkdir(exist_ok=True)
        for name in COMPUTE_FILES:
            shutil.copyfile(CODE / name, work_code / name)
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        if args.mode == "all":
            run_optimization(work_code, results, args.workers, args.reuse_ledger)
        run_simulations(work_code, results, output, args.workers)

    regenerate_exhibits(results, output)
    print(f"Outputs written to {output}")


if __name__ == "__main__":
    main()
