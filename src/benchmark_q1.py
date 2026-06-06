"""
Question 1 Benchmark — run Method 1 (pure Python loops) N times
on the current container and save per-run execution times.

Usage (inside Docker or locally):
    python src/benchmark_q1.py --computer computer1
    python src/benchmark_q1.py --computer computer2 --runs 50 --warmup 3
"""

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from assignment_methods import method1_python_loops_parallel, _get_container_cpu_count
from data_loader import load_both
from system_info import get_system_info
from utils import get_data_dir, get_results_dir

N_WARMUP = 3
N_RUNS = 50


def _print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def save_system_specs(computer_name: str, results_dir: str) -> None:
    """Collect and persist container/hardware specs to CSV."""
    info = get_system_info()
    info["computer_name"] = computer_name

    specs_path = os.path.join(results_dir, "q1_system_specs.csv")

    if os.path.exists(specs_path):
        existing = pd.read_csv(specs_path)
        # Replace any previous entry for this computer
        existing = existing[existing["computer_name"] != computer_name]
        specs_df = pd.concat([existing, pd.DataFrame([info])], ignore_index=True)
    else:
        specs_df = pd.DataFrame([info])

    specs_df.to_csv(specs_path, index=False)

    print("\n[Specs]")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print(f"\nSpecs saved -> {specs_path}")


def run_benchmark(
    computer_name: str,
    n_runs: int = N_RUNS,
    n_warmup: int = N_WARMUP,
) -> list[float]:
    """
    Execute the Q1 benchmark:
      1. Save system specs.
      2. Perform n_warmup unmeasured warm-up runs (allows the OS/interpreter
         to page in code and data so the first real run is not penalised).
      3. Perform n_runs measured runs; each run reloads the CSV files and
         executes the full assignment algorithm.
      4. Save per-run times to results/q1/<computer_name>_times.csv.
      5. Save a 100-row assignment sample.
    """
    results_dir = get_results_dir("q1")
    data_dir = get_data_dir()

    _print_header(f"Q1 Benchmark — {computer_name}")
    save_system_specs(computer_name, results_dir)

    n_workers = _get_container_cpu_count()
    print(f"\n[Parallelism] Using {n_workers} worker process(es) (from cgroup CPU quota).")
    print( "[Note] Each worker runs the same Haversine loop on its share of people.")
    print(f"       computer1 (4 CPUs) -> 4 workers -> ~4x faster than single-core")
    print(f"       computer2 (1 CPU)  -> 1 worker  -> same as sequential\n")

    # ── warm-up ──────────────────────────────────────────────────────────
    print(f"\nWarm-up ({n_warmup} run(s), not timed) ...")
    for i in range(n_warmup):
        p, c = load_both(data_dir)
        _ = method1_python_loops_parallel(p, c)
        print(f"  warm-up {i + 1}/{n_warmup} done")

    # ── measured runs ─────────────────────────────────────────────────────
    print(f"\nMeasured runs ({n_runs}) ...")
    times: list[float] = []
    last_result = None

    for i in range(n_runs):
        # Reload CSV inside the timed section (as required)
        t_start = time.perf_counter()
        p, c = load_both(data_dir)
        result_df = method1_python_loops_parallel(p, c)
        t_end = time.perf_counter()

        elapsed = t_end - t_start
        times.append(elapsed)
        last_result = result_df

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  run {i + 1:>3}/{n_runs}  time = {elapsed:.4f} s")

    # ── persist results ───────────────────────────────────────────────────
    times_df = pd.DataFrame(
        {
            "run": range(1, n_runs + 1),
            "execution_time_s": times,
            "computer": computer_name,
        }
    )
    times_path = os.path.join(results_dir, f"{computer_name}_times.csv")
    times_df.to_csv(times_path, index=False)
    print(f"\nTiming data saved -> {times_path}")

    if last_result is not None:
        sample_path = os.path.join(results_dir, "q1_assignment_sample.csv")
        last_result.head(100).to_csv(sample_path, index=False)
        print(f"Assignment sample saved -> {sample_path}")

    mean_t = sum(times) / len(times)
    print(
        f"\nSummary  |  mean={mean_t:.4f}s  "
        f"min={min(times):.4f}s  max={max(times):.4f}s"
    )
    return times


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q1 Benchmark")
    parser.add_argument(
        "--computer",
        type=str,
        default="computer1",
        choices=["computer1", "computer2"],
        help="Container identifier (used as file-name prefix)",
    )
    parser.add_argument("--runs", type=int, default=N_RUNS)
    parser.add_argument("--warmup", type=int, default=N_WARMUP)
    args = parser.parse_args()

    run_benchmark(args.computer, args.runs, args.warmup)
