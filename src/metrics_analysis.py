# src/metrics_analysis.py
# Descriptive statistics and metric definitions for software quality metrics.

import pandas as pd
import numpy as np

# Official definitions for each supported metric.
METRIC_DEFINITIONS = {
    "LOC": {
        "name": "Lines of Code",
        "unit": "Count (lines)",
        "description": (
            "Total number of source lines of code in the module, "
            "excluding blank lines and comments (varies by convention)."
        ),
        "interpretation": (
            "Higher LOC generally indicates larger, more complex modules that "
            "may be harder to maintain. However, LOC alone does not indicate quality."
        ),
        "threshold_warning": 500,
    },
    "CC": {
        "name": "Cyclomatic Complexity",
        "unit": "Count (paths)",
        "description": (
            "McCabe's cyclomatic complexity: the number of linearly independent "
            "paths through the module's control-flow graph. "
            "CC = E − N + 2P, where E=edges, N=nodes, P=connected components."
        ),
        "interpretation": (
            "CC ≤ 10 is generally considered low risk. "
            "CC 11–20 is moderate. CC > 20 is high risk and hard to test."
        ),
        "threshold_warning": 10,
    },
    "Dup": {
        "name": "Code Duplication (%)",
        "unit": "Percentage (0–100)",
        "description": (
            "Percentage of the module's code that is duplicated elsewhere "
            "in the codebase, as reported by a static analysis tool."
        ),
        "interpretation": (
            "High duplication (>15–20%) increases maintenance cost and bug risk "
            "because fixes must be applied in multiple places."
        ),
        "threshold_warning": 15,
    },
    "Churn": {
        "name": "Code Churn",
        "unit": "Count (changes in window)",
        "description": (
            "Number of times the module was modified (commits, edits) during "
            "a defined time window, as recorded by the version-control system."
        ),
        "interpretation": (
            "High churn can indicate instability or ongoing refactoring. "
            "Combined with high complexity it is a strong debt indicator."
        ),
        "threshold_warning": 30,
    },
    "Coupling": {
        "name": "Coupling",
        "unit": "Count (dependencies)",
        "description": (
            "Number of other modules that this module directly depends on "
            "(afferent + efferent coupling)."
        ),
        "interpretation": (
            "High coupling (>10) makes modules fragile: changing one module "
            "ripples to many others and complicates testing."
        ),
        "threshold_warning": 8,
    },
    "Maintainability": {
        "name": "Maintainability Index",
        "unit": "Score (0–100)",
        "description": (
            "A composite score, often computed from LOC, CC and Halstead volume "
            "(MI = 171 − 5.2·ln(V) − 0.23·CC − 16.2·ln(LOC), clamped to 0–100). "
            "This application accepts pre-computed values only; "
            "raw source code is not analysed."
        ),
        "interpretation": (
            "MI ≥ 85 is considered highly maintainable; 65–85 moderate; "
            "<65 low maintainability."
        ),
        "threshold_warning": 65,
    },
}


def compute_descriptive_stats(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Return a DataFrame of descriptive stats for the specified columns."""
    available = [c for c in columns if c in df.columns]
    if not available:
        return pd.DataFrame()
    stats = df[available].describe(percentiles=[0.25, 0.5, 0.75]).T
    stats.index.name = "Metric"
    stats.columns = [c.capitalize() for c in stats.columns]
    return stats.reset_index()


def compute_heuristic_risk(df: pd.DataFrame) -> pd.Series:
    """Compute a normalised 0-100 heuristic risk score per row.

    Formula (documented):
        score = 0.30·norm(LOC) + 0.30·norm(CC) + 0.15·norm(Dup)
                + 0.15·norm(Churn) + 0.10·norm(Coupling)

    Each metric is min-max normalised to [0, 1].
    Missing metrics are excluded and weights are re-scaled to sum to 1.

    IMPORTANT: This is a heuristic estimate — NOT a supervised ML prediction.
    """
    weights = {
        "LOC": 0.30,
        "CC": 0.30,
        "Dup": 0.15,
        "Churn": 0.15,
        "Coupling": 0.10,
    }
    present = {col: w for col, w in weights.items() if col in df.columns}
    if not present:
        return pd.Series([np.nan] * len(df), index=df.index)

    total_weight = sum(present.values())
    score = pd.Series(0.0, index=df.index)

    for col, w in present.items():
        col_data = df[col].copy()
        col_min, col_max = col_data.min(), col_data.max()
        if col_max > col_min:
            normed = (col_data - col_min) / (col_max - col_min)
        else:
            normed = pd.Series(0.0, index=df.index)
        score += (w / total_weight) * normed

    return (score * 100).round(1)


def flag_outliers_iqr(series: pd.Series) -> pd.Series:
    """Return boolean Series: True where value is an IQR outlier (1.5×IQR rule)."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper)
