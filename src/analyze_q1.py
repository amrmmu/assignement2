"""
Question 1 Analysis — compare execution-time distributions between
Computer 1 (4 CPUs, 4 GB) and Computer 2 (1 CPU, 512 MB).

Outputs (written to results/q1/):
  table_1_execution_time_comparison.csv
  q1_summary_statistics.csv
  q1_statistical_tests.csv
  q1_density_plot.png
  q1_boxplot.png

Statistical decision tree:
  1. Shapiro-Wilk normality test on each group.
  2a. Both normal   → Welch's independent t-test.
  2b. Either non-normal → Mann-Whitney U test.
  3. Report p-value, significance at alpha = 0.05, and which
     computer has the lower mean execution time.
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")          # non-interactive backend; must be before pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import get_results_dir

ALPHA = 0.05

# ── palette ───────────────────────────────────────────────────────────────────
C1_COLOR = "steelblue"
C2_COLOR = "coral"
C1_LABEL = "Computer 1 (4 CPUs, 4 GB)"
C2_LABEL = "Computer 2 (1 CPU, 512 MB)"


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_times(results_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load per-run execution times for both computers."""
    for fname in ("computer1_times.csv", "computer2_times.csv"):
        path = os.path.join(results_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Run the Q1 benchmarks first:\n"
                "  docker compose run --rm computer1\n"
                "  docker compose run --rm computer2"
            )
    c1 = pd.read_csv(os.path.join(results_dir, "computer1_times.csv"))[
        "execution_time_s"
    ].values
    c2 = pd.read_csv(os.path.join(results_dir, "computer2_times.csv"))[
        "execution_time_s"
    ].values
    return c1, c2


def _summary_stats(data: np.ndarray, label: str) -> dict:
    return {
        "group": label,
        "n": len(data),
        "mean": np.mean(data),
        "median": np.median(data),
        "std": np.std(data, ddof=1),
        "min": np.min(data),
        "max": np.max(data),
        "q25": np.percentile(data, 25),
        "q75": np.percentile(data, 75),
    }


# ── table / stats ─────────────────────────────────────────────────────────────

def generate_table1(
    c1: np.ndarray, c2: np.ndarray, results_dir: str
) -> pd.DataFrame:
    """Table 1: side-by-side execution times (columns: No, Computer 1, Computer 2)."""
    n = max(len(c1), len(c2))
    table = pd.DataFrame(
        {
            "No": range(1, n + 1),
            "Computer 1 (s)": list(c1) + [float("nan")] * (n - len(c1)),
            "Computer 2 (s)": list(c2) + [float("nan")] * (n - len(c2)),
        }
    )
    path = os.path.join(results_dir, "table_1_execution_time_comparison.csv")
    table.to_csv(path, index=False)
    print(f"Table 1 saved -> {path}")
    return table


def generate_summary_stats(
    c1: np.ndarray, c2: np.ndarray, results_dir: str
) -> pd.DataFrame:
    summary = pd.DataFrame(
        [_summary_stats(c1, "Computer 1"), _summary_stats(c2, "Computer 2")]
    )
    path = os.path.join(results_dir, "q1_summary_statistics.csv")
    summary.to_csv(path, index=False)
    print(f"Summary statistics saved -> {path}")
    return summary


def run_statistical_tests(
    c1: np.ndarray, c2: np.ndarray, results_dir: str
) -> dict:
    """
    1. Shapiro-Wilk for each group.
    2. Welch t-test if both normal; Mann-Whitney U otherwise.
    3. Print and save results.
    """
    print(f"\n--- Normality Tests (Shapiro-Wilk, alpha={ALPHA}) ---")

    sw_stat1, sw_p1 = stats.shapiro(c1)
    sw_stat2, sw_p2 = stats.shapiro(c2)
    c1_normal = sw_p1 > ALPHA
    c2_normal = sw_p2 > ALPHA

    print(
        f"  Computer 1: W={sw_stat1:.4f}, p={sw_p1:.6f} "
        f"-> {'NORMAL' if c1_normal else 'NOT NORMAL'}"
    )
    print(
        f"  Computer 2: W={sw_stat2:.4f}, p={sw_p2:.6f} "
        f"-> {'NORMAL' if c2_normal else 'NOT NORMAL'}"
    )

    both_normal = c1_normal and c2_normal

    if both_normal:
        test_stat, p_value = stats.ttest_ind(c1, c2, equal_var=False)
        test_name = "Welch's Independent t-test"
    else:
        test_stat, p_value = stats.mannwhitneyu(c1, c2, alternative="two-sided")
        test_name = "Mann-Whitney U test"

    significant = p_value < ALPHA
    faster = "Computer 1" if np.mean(c1) < np.mean(c2) else "Computer 2"
    slower = "Computer 2" if faster == "Computer 1" else "Computer 1"
    faster_mean = np.mean(c1) if faster == "Computer 1" else np.mean(c2)
    slower_mean = np.mean(c2) if faster == "Computer 1" else np.mean(c1)

    print(f"\n--- {test_name} ---")
    print(f"  test statistic = {test_stat:.4f}")
    print(f"  p-value        = {p_value:.6f}")
    print(f"  significant (alpha={ALPHA}): {significant}")
    print(f"  faster computer: {faster}  (mean={faster_mean:.4f}s)")
    print(f"  slower computer: {slower}  (mean={slower_mean:.4f}s)")

    interpretation = (
        f"The {test_name} yielded a test statistic of {test_stat:.4f} and "
        f"a p-value of {p_value:.6f}. "
        f"At a significance level of alpha={ALPHA}, the difference in execution time "
        f"between the two computers is "
        f"{'STATISTICALLY SIGNIFICANT' if significant else 'NOT statistically significant'}. "
        f"{faster} is faster with a mean execution time of {faster_mean:.4f} s "
        f"compared to {slower_mean:.4f} s for {slower}."
    )
    print(f"\nInterpretation:\n  {interpretation}")

    row = {
        "shapiro_W_c1": sw_stat1, "shapiro_p_c1": sw_p1,
        "shapiro_W_c2": sw_stat2, "shapiro_p_c2": sw_p2,
        "c1_normal": c1_normal, "c2_normal": c2_normal,
        "test_used": test_name,
        "test_statistic": test_stat,
        "p_value": p_value,
        "alpha": ALPHA,
        "significant": significant,
        "faster_computer": faster,
        "mean_c1_s": np.mean(c1),
        "mean_c2_s": np.mean(c2),
        "interpretation": interpretation,
    }
    pd.DataFrame([row]).to_csv(
        os.path.join(results_dir, "q1_statistical_tests.csv"), index=False
    )
    print(f"Statistical tests saved -> {os.path.join(results_dir, 'q1_statistical_tests.csv')}")
    return row


# ── plots ─────────────────────────────────────────────────────────────────────

def create_density_plot(
    c1: np.ndarray, c2: np.ndarray, results_dir: str
) -> None:
    """Shaded KDE (density) plot comparing both computers."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Main KDE fills
    sns.kdeplot(
        c1, ax=ax, fill=True, alpha=0.35,
        color=C1_COLOR, label=C1_LABEL, common_norm=False,
    )
    sns.kdeplot(
        c2, ax=ax, fill=True, alpha=0.35,
        color=C2_COLOR, label=C2_LABEL, common_norm=False,
    )

    # Mean lines
    ax.axvline(
        np.mean(c1), color=C1_COLOR, linestyle="--", linewidth=1.5,
        label=f"C1 mean = {np.mean(c1):.4f} s",
    )
    ax.axvline(
        np.mean(c2), color=C2_COLOR, linestyle="--", linewidth=1.5,
        label=f"C2 mean = {np.mean(c2):.4f} s",
    )

    ax.set_xlabel("Execution Time (seconds)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(
        "Q1: Execution Time Distribution\nComputer 1 (4 CPUs, 4 GB) vs "
        "Computer 2 (1 CPU, 512 MB)",
        fontsize=14,
    )
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    sns.despine()
    plt.tight_layout()

    path = os.path.join(results_dir, "q1_density_plot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Density plot saved -> {path}")


def create_boxplot(
    c1: np.ndarray, c2: np.ndarray, results_dir: str
) -> None:
    """Boxplot with overlaid jitter comparing both computers."""
    df = pd.DataFrame(
        {
            "Execution Time (s)": np.concatenate([c1, c2]),
            "Computer": [C1_LABEL] * len(c1) + [C2_LABEL] * len(c2),
        }
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        data=df, x="Computer", y="Execution Time (s)",
        hue="Computer", palette=[C1_COLOR, C2_COLOR],
        ax=ax, width=0.5, legend=False,
    )
    sns.stripplot(
        data=df, x="Computer", y="Execution Time (s)",
        color="black", alpha=0.35, size=3, jitter=True, ax=ax,
    )

    ax.set_title(
        "Q1: Execution Time Comparison\nComputer 1 vs Computer 2", fontsize=14
    )
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.3)
    sns.despine()
    plt.tight_layout()

    path = os.path.join(results_dir, "q1_boxplot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Boxplot saved -> {path}")


# ── main entry point ──────────────────────────────────────────────────────────

def run_analysis() -> dict:
    """Run the complete Q1 analysis pipeline."""
    results_dir = get_results_dir("q1")
    print(f"\n{'=' * 60}")
    print("  Q1 Analysis: Computer 1 vs Computer 2")
    print(f"{'=' * 60}")

    c1, c2 = _load_times(results_dir)
    print(f"Loaded  Computer 1: {len(c1)} runs  |  Computer 2: {len(c2)} runs")

    generate_table1(c1, c2, results_dir)
    generate_summary_stats(c1, c2, results_dir)
    test_results = run_statistical_tests(c1, c2, results_dir)
    create_density_plot(c1, c2, results_dir)
    create_boxplot(c1, c2, results_dir)

    print("\nQ1 analysis complete!")
    return test_results


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    run_analysis()
