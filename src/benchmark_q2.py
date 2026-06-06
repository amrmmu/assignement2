"""
Question 2 Benchmark — run all three assignment methods N times
on the current container and save per-run execution times.

Usage (inside Docker or locally):
    python src/benchmark_q2.py
    python src/benchmark_q2.py --runs 50 --warmup 3
"""

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from assignment_methods import METHODS
from data_loader import load_both
from utils import get_data_dir, get_results_dir

N_WARMUP = 3
N_RUNS = 50


def _print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def benchmark_one_method(
    method_key: str,
    method_name: str,
    method_fn,
    data_dir: str,
    results_dir: str,
    n_runs: int,
    n_warmup: int,
) -> tuple[list[float], pd.DataFrame | None]:
    """
    Benchmark a single assignment method.

    Returns (list_of_times, last_result_df).
    """
    _print_header(f"Benchmarking {method_name}  [{method_key}]")

    # ── warm-up ──────────────────────────────────────────────────────────
    print(f"Warm-up ({n_warmup} run(s), not timed) ...")
    for i in range(n_warmup):
        p, c = load_both(data_dir)
        _ = method_fn(p, c)
        print(f"  warm-up {i + 1}/{n_warmup} done")

    # ── measured runs ─────────────────────────────────────────────────────
    print(f"Measured runs ({n_runs}) ...")
    times: list[float] = []
    last_result = None

    for i in range(n_runs):
        t_start = time.perf_counter()
        p, c = load_both(data_dir)
        result_df = method_fn(p, c)
        t_end = time.perf_counter()

        elapsed = t_end - t_start
        times.append(elapsed)
        last_result = result_df

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  run {i + 1:>3}/{n_runs}  time = {elapsed:.4f} s")

    # ── persist ───────────────────────────────────────────────────────────
    times_df = pd.DataFrame(
        {
            "run": range(1, n_runs + 1),
            "execution_time_s": times,
            "method": method_key,
        }
    )
    times_path = os.path.join(results_dir, f"{method_key}_times.csv")
    times_df.to_csv(times_path, index=False)
    print(f"Timing data saved -> {times_path}")

    mean_t = sum(times) / len(times)
    print(
        f"Summary  |  mean={mean_t:.4f}s  "
        f"min={min(times):.4f}s  max={max(times):.4f}s"
    )
    return times, last_result


def run_all_benchmarks(n_runs: int = N_RUNS, n_warmup: int = N_WARMUP) -> None:
    """Run benchmarks for all three methods sequentially."""
    results_dir = get_results_dir("q2")
    data_dir = get_data_dir()

    _print_header("Q2 Benchmark — 3 Assignment Methods")

    last_result = None
    for method_key, (method_name, method_fn) in METHODS.items():
        _, result = benchmark_one_method(
            method_key, method_name, method_fn,
            data_dir, results_dir, n_runs, n_warmup,
        )
        if result is not None:
            last_result = result

    if last_result is not None:
        sample_path = os.path.join(results_dir, "q2_assignment_sample.csv")
        last_result.head(100).to_csv(sample_path, index=False)
        print(f"\nAssignment sample saved -> {sample_path}")

    print("\nAll Q2 benchmarks complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q2 Benchmark")
    parser.add_argument("--runs", type=int, default=N_RUNS)
    parser.add_argument("--warmup", type=int, default=N_WARMUP)
    args = parser.parse_args()

    run_all_benchmarks(args.runs, args.warmup)
