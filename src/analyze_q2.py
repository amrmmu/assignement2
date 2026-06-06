"""
Question 2 Analysis — compare execution-time distributions across
three assignment methods.

Outputs (written to results/q2/):
  table_2_execution_time_comparison.csv
  q2_summary_statistics.csv
  q2_statistical_tests.csv
  q2_posthoc_tests.csv
  q2_density_plot.png
  q2_boxplot.png

Statistical decision tree:
  1. Shapiro-Wilk normality test on each method group.
  2a. All normal   → one-way ANOVA.
  2b. Any non-normal → Kruskal-Wallis test.
  3. If overall test is significant:
     3a. All normal   → pairwise Welch t-tests + Bonferroni correction.
     3b. Any non-normal → pairwise Mann-Whitney U + Bonferroni correction.
  4. Report p-values, significance at alpha = 0.05, and method ranking.
"""

import os
import sys
import warnings
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import get_results_dir

ALPHA = 0.05
METHOD_KEYS = ["method1", "method2", "method3"]
METHOD_DISPLAY = {
    "method1": "Method 1\n(Python Loops)",
    "method2": "Method 2\n(NumPy)",
    "method3": "Method 3\n(cKDTree)",
}
METHOD_LEGEND = {
    "method1": "Method 1 — Python Loops",
    "method2": "Method 2 — NumPy Broadcasting",
    "method3": "Method 3 — cKDTree Spatial Index",
}
PALETTE = ["steelblue", "coral", "seagreen"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_times(results_dir: str) -> dict[str, np.ndarray]:
    """Load per-run execution times for all three methods."""
    times = {}
    for key in METHOD_KEYS:
        path = os.path.join(results_dir, f"{key}_times.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Run the Q2 benchmark first:\n"
                "  docker compose run --rm q2_methods"
            )
        times[key] = pd.read_csv(path)["execution_time_s"].values
    return times


def _summary_stats(data: np.ndarray, label: str) -> dict:
    return {
        "method": label,
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

def generate_table2(
    times: dict[str, np.ndarray], results_dir: str
) -> pd.DataFrame:
    """Table 2: side-by-side execution times for all three methods."""
    n = max(len(v) for v in times.values())
    table = pd.DataFrame({"No": range(1, n + 1)})
    for key, col_label in [
        ("method1", "Method 1 (s)"),
        ("method2", "Method 2 (s)"),
        ("method3", "Method 3 (s)"),
    ]:
        data = times[key]
        table[col_label] = list(data) + [float("nan")] * (n - len(data))
    path = os.path.join(results_dir, "table_2_execution_time_comparison.csv")
    table.to_csv(path, index=False)
    print(f"Table 2 saved -> {path}")
    return table


def generate_summary_stats(
    times: dict[str, np.ndarray], results_dir: str
) -> pd.DataFrame:
    rows = [_summary_stats(v, k) for k, v in times.items()]
    df = pd.DataFrame(rows)
    path = os.path.join(results_dir, "q2_summary_statistics.csv")
    df.to_csv(path, index=False)
    print(f"Summary statistics saved -> {path}")
    return df


def run_statistical_tests(
    times: dict[str, np.ndarray], results_dir: str
) -> dict:
    """
    Run normality tests, choose overall test, run post-hoc if significant.
    """
    print(f"\n--- Normality Tests (Shapiro-Wilk, alpha={ALPHA}) ---")
    normality: dict[str, dict] = {}
    for key, data in times.items():
        w, p = stats.shapiro(data)
        is_normal = p > ALPHA
        normality[key] = {"W": w, "p": p, "normal": is_normal}
        print(
            f"  {key}: W={w:.4f}, p={p:.6f} "
            f"-> {'NORMAL' if is_normal else 'NOT NORMAL'}"
        )

    all_normal = all(v["normal"] for v in normality.values())

    # ── overall test ──────────────────────────────────────────────────────
    groups = list(times.values())
    if all_normal:
        test_stat, p_value = stats.f_oneway(*groups)
        overall_test = "One-way ANOVA"
    else:
        test_stat, p_value = stats.kruskal(*groups)
        overall_test = "Kruskal-Wallis H test"

    significant = p_value < ALPHA

    print(f"\n--- Overall Test: {overall_test} ---")
    print(f"  statistic = {test_stat:.4f}")
    print(f"  p-value   = {p_value:.6f}")
    print(f"  significant (alpha={ALPHA}): {significant}")

    # Rank methods by mean
    ranked = sorted(times.items(), key=lambda kv: np.mean(kv[1]))
    print("\n  Ranking (fastest -> slowest):")
    for rank, (key, data) in enumerate(ranked, 1):
        print(f"    {rank}. {key}  mean={np.mean(data):.4f} s")

    # ── post-hoc pairwise tests ───────────────────────────────────────────
    posthoc_rows: list[dict] = []
    if significant:
        pairs = list(combinations(list(times.keys()), 2))
        n_comp = len(pairs)
        bonf_alpha = ALPHA / n_comp
        print(
            f"\n--- Post-hoc pairwise tests "
            f"(Bonferroni-corrected alpha = {bonf_alpha:.4f}) ---"
        )
        for k1, k2 in pairs:
            if all_normal:
                pstat, p_raw = stats.ttest_ind(
                    times[k1], times[k2], equal_var=False
                )
                test_label = "Welch t-test"
            else:
                pstat, p_raw = stats.mannwhitneyu(
                    times[k1], times[k2], alternative="two-sided"
                )
                test_label = "Mann-Whitney U"
            p_bonf = min(p_raw * n_comp, 1.0)
            sig = p_bonf < ALPHA
            print(
                f"  {k1} vs {k2}: stat={pstat:.4f}  "
                f"p_raw={p_raw:.6f}  p_bonf={p_bonf:.6f}  sig={sig}"
            )
            posthoc_rows.append(
                {
                    "pair": f"{k1} vs {k2}",
                    "test": test_label,
                    "statistic": pstat,
                    "p_raw": p_raw,
                    "p_bonferroni": p_bonf,
                    "significant_after_correction": sig,
                }
            )

    posthoc_df = pd.DataFrame(posthoc_rows) if posthoc_rows else pd.DataFrame(
        columns=["pair", "test", "statistic", "p_raw",
                 "p_bonferroni", "significant_after_correction"]
    )
    ph_path = os.path.join(results_dir, "q2_posthoc_tests.csv")
    posthoc_df.to_csv(ph_path, index=False)
    print(f"Post-hoc tests saved -> {ph_path}")

    # ── build result row ──────────────────────────────────────────────────
    interpretation = (
        f"The {overall_test} yielded a test statistic of {test_stat:.4f} and "
        f"a p-value of {p_value:.6f}. "
        f"At alpha={ALPHA}, the differences in execution time among the three "
        f"methods are "
        f"{'STATISTICALLY SIGNIFICANT' if significant else 'NOT statistically significant'}. "
        f"Ranking from fastest to slowest: "
        + ", ".join(
            f"{k} (mean={np.mean(v):.4f} s)" for k, v in ranked
        )
        + "."
    )
    print(f"\nInterpretation:\n  {interpretation}")

    row: dict = {
        "overall_test": overall_test,
        "test_statistic": test_stat,
        "p_value": p_value,
        "alpha": ALPHA,
        "significant": significant,
        "all_normal": all_normal,
        "ranking_fastest_to_slowest": " > ".join(k for k, _ in ranked),
        "interpretation": interpretation,
    }
    for key, nd in normality.items():
        row[f"shapiro_W_{key}"] = nd["W"]
        row[f"shapiro_p_{key}"] = nd["p"]
        row[f"normal_{key}"] = nd["normal"]

    pd.DataFrame([row]).to_csv(
        os.path.join(results_dir, "q2_statistical_tests.csv"), index=False
    )
    print(
        f"Statistical tests saved -> "
        f"{os.path.join(results_dir, 'q2_statistical_tests.csv')}"
    )
    return row


# ── plots ─────────────────────────────────────────────────────────────────────

def create_density_plot(
    times: dict[str, np.ndarray], results_dir: str
) -> None:
    """Shaded KDE density plot for all three methods."""
    fig, ax = plt.subplots(figsize=(11, 6))

    for (key, data), color in zip(times.items(), PALETTE):
        label = METHOD_LEGEND[key]
        sns.kdeplot(
            data, ax=ax, fill=True, alpha=0.35,
            color=color, label=label, common_norm=False,
        )
        ax.axvline(
            np.mean(data), color=color, linestyle="--", linewidth=1.5,
            label=f"{label.split(' — ')[0]} mean = {np.mean(data):.4f} s",
        )

    ax.set_xlabel("Execution Time (seconds)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(
        "Q2: Execution Time Distribution\n"
        "Method 1 (Python Loops) vs Method 2 (NumPy) vs Method 3 (cKDTree)",
        fontsize=14,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    sns.despine()
    plt.tight_layout()

    path = os.path.join(results_dir, "q2_density_plot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Density plot saved -> {path}")

    # ── log-scale version (helpful when Method 1 is much slower) ─────────
    fig2, ax2 = plt.subplots(figsize=(11, 6))
    for (key, data), color in zip(times.items(), PALETTE):
        label = METHOD_LEGEND[key]
        log_data = np.log10(data)
        sns.kdeplot(
            log_data, ax=ax2, fill=True, alpha=0.35,
            color=color, label=label, common_norm=False,
        )
        ax2.axvline(
            np.log10(np.mean(data)), color=color, linestyle="--", linewidth=1.5,
        )
    ax2.set_xlabel("log10(Execution Time [s])", fontsize=13)
    ax2.set_ylabel("Density", fontsize=13)
    ax2.set_title(
        "Q2: Execution Time Distribution (log10 scale)\n"
        "Method 1 vs Method 2 vs Method 3",
        fontsize=14,
    )
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    sns.despine()
    plt.tight_layout()

    path2 = os.path.join(results_dir, "q2_density_plot_log.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Log-scale density plot saved -> {path2}")


def create_boxplot(
    times: dict[str, np.ndarray], results_dir: str
) -> None:
    """Boxplot with overlaid jitter for all three methods."""
    rows = []
    for key, data in times.items():
        for val in data:
            rows.append(
                {"Execution Time (s)": val, "Method": METHOD_DISPLAY[key]}
            )
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    x_order = [METHOD_DISPLAY[k] for k in METHOD_KEYS]
    sns.boxplot(
        data=df, x="Method", y="Execution Time (s)",
        order=x_order, hue="Method", palette=PALETTE,
        ax=ax, width=0.5, legend=False,
    )
    sns.stripplot(
        data=df, x="Method", y="Execution Time (s)",
        order=x_order, color="black", alpha=0.3, size=3, jitter=True, ax=ax,
    )
    ax.set_title(
        "Q2: Execution Time Comparison\n3 Assignment Methods", fontsize=14
    )
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.3)
    sns.despine()
    plt.tight_layout()

    path = os.path.join(results_dir, "q2_boxplot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Boxplot saved -> {path}")


# ── main entry point ──────────────────────────────────────────────────────────

def run_analysis() -> dict:
    """Run the complete Q2 analysis pipeline."""
    results_dir = get_results_dir("q2")
    print(f"\n{'=' * 60}")
    print("  Q2 Analysis: Method 1 vs Method 2 vs Method 3")
    print(f"{'=' * 60}")

    times = _load_times(results_dir)
    for key, data in times.items():
        print(f"Loaded  {key}: {len(data)} runs")

    generate_table2(times, results_dir)
    generate_summary_stats(times, results_dir)
    test_results = run_statistical_tests(times, results_dir)
    create_density_plot(times, results_dir)
    create_boxplot(times, results_dir)

    print("\nQ2 analysis complete!")
    return test_results


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    run_analysis()
