# src/statistics_analysis.py
# Statistical tests, correlations, and grouped comparisons.

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats
import io


def pearson_correlation(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Compute Pearson correlation matrix for numeric columns."""
    available = [c for c in columns if c in df.columns]
    if len(available) < 2:
        return pd.DataFrame()
    sub = df[available].select_dtypes(include="number").dropna()
    # Drop constant columns (std == 0)
    sub = sub.loc[:, sub.std() > 0]
    if sub.shape[1] < 2:
        return pd.DataFrame()
    return sub.corr(method="pearson")


def spearman_correlation(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Compute Spearman rank correlation matrix for numeric columns."""
    available = [c for c in columns if c in df.columns]
    if len(available) < 2:
        return pd.DataFrame()
    sub = df[available].select_dtypes(include="number").dropna()
    sub = sub.loc[:, sub.std() > 0]
    if sub.shape[1] < 2:
        return pd.DataFrame()
    return sub.corr(method="spearman")


def mann_whitney_test(df: pd.DataFrame, metric: str, group_col: str = "Risk") -> dict:
    """Compare High vs Low risk groups using Mann-Whitney U test.

    Returns a dict with sample sizes, statistic, p_value, and interpretation.
    Requires at least 3 observations per group.
    """
    if metric not in df.columns or group_col not in df.columns:
        return {"error": "Required column(s) not present."}

    high = df[df[group_col] == "High"][metric].dropna()
    low = df[df[group_col] == "Low"][metric].dropna()

    if len(high) < 3 or len(low) < 3:
        return {
            "error": (
                f"Insufficient data: High group n={len(high)}, Low group n={len(low)}. "
                "At least 3 observations per group are required."
            )
        }

    stat, p_value = scipy_stats.mannwhitneyu(high, low, alternative="two-sided")
    alpha = 0.05
    significant = p_value < alpha
    return {
        "metric": metric,
        "n_high": int(len(high)),
        "n_low": int(len(low)),
        "mean_high": float(high.mean()),
        "mean_low": float(low.mean()),
        "median_high": float(high.median()),
        "median_low": float(low.median()),
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": significant,
        "interpretation": (
            f"p = {p_value:.4f}. "
            + (
                f"There is a statistically significant difference (α=0.05) in '{metric}' "
                "between High and Low risk groups."
                if significant
                else f"No statistically significant difference detected (α=0.05) in '{metric}'."
            )
            + " Note: statistical significance depends on sample size and "
            "does not prove causation."
        ),
    }


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, title: str = "Correlation Matrix") -> bytes:
    """Return PNG bytes of a seaborn correlation heatmap."""
    if corr_matrix.empty:
        return b""
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def plot_boxplot(
    df: pd.DataFrame,
    metric: str,
    group_col: str | None = None,
) -> bytes:
    """Return PNG bytes of a box plot for the given metric, optionally grouped."""
    if metric not in df.columns:
        return b""
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {"High": "#E05252", "Low": "#52A8E0"}

    if group_col and group_col in df.columns:
        order = ["Low", "High"]
        groups_present = [g for g in order if g in df[group_col].values]
        sns.boxplot(
            data=df,
            x=group_col,
            y=metric,
            order=groups_present,
            palette=palette,
            ax=ax,
            width=0.5,
            flierprops=dict(marker="o", markerfacecolor="orange", markersize=6),
        )
        ax.set_xlabel("Risk Group", fontsize=11)
    else:
        sns.boxplot(
            data=df,
            y=metric,
            ax=ax,
            color="#5B8DB8",
            width=0.4,
            flierprops=dict(marker="o", markerfacecolor="orange", markersize=6),
        )
        ax.set_xlabel("")

    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f"Box Plot — {metric}", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
