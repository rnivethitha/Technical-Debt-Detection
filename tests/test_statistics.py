# tests/test_statistics.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np
from src.statistics_analysis import (
    pearson_correlation, spearman_correlation, mann_whitney_test
)
from src.metrics_analysis import (
    compute_descriptive_stats, compute_heuristic_risk, flag_outliers_iqr
)
from src.control_chart import compute_imr_chart


def _make_df(n=30):
    rng = np.random.default_rng(42)
    high = int(n * 0.5)
    low = n - high
    return pd.DataFrame({
        "Module": [f"M{i}" for i in range(n)],
        "LOC":    list(rng.integers(100, 800, n)),
        "CC":     list(rng.integers(3, 25, n)),
        "Dup":    list(rng.uniform(2, 35, n)),
        "Churn":  list(rng.integers(5, 70, n)),
        "Coupling": list(rng.integers(2, 16, n)),
        "Risk":   ["High"] * high + ["Low"] * low,
    })


# ── Descriptive stats ────────────────────────────────────────────────────
def test_descriptive_stats_returns_dataframe():
    df = _make_df()
    stats = compute_descriptive_stats(df, ["LOC", "CC"])
    assert not stats.empty
    assert "Metric" in stats.columns


def test_descriptive_stats_empty_for_missing_col():
    df = _make_df()
    stats = compute_descriptive_stats(df, ["NonExistent"])
    assert stats.empty


# ── Pearson / Spearman ───────────────────────────────────────────────────
def test_pearson_returns_square_matrix():
    df = _make_df()
    corr = pearson_correlation(df, ["LOC", "CC", "Dup"])
    assert corr.shape == (3, 3)


def test_spearman_returns_square_matrix():
    df = _make_df()
    corr = spearman_correlation(df, ["LOC", "CC"])
    assert corr.shape == (2, 2)


def test_pearson_returns_empty_for_insufficient_cols():
    df = _make_df()
    corr = pearson_correlation(df, ["LOC"])
    assert corr.empty


def test_pearson_handles_constant_column():
    df = _make_df()
    df["Constant"] = 5
    corr = pearson_correlation(df, ["LOC", "Constant"])
    # Constant column should be dropped; result should be empty or 1×1
    assert corr.empty or corr.shape[0] <= 1


# ── Mann-Whitney ─────────────────────────────────────────────────────────
def test_mann_whitney_returns_result():
    df = _make_df(n=30)
    result = mann_whitney_test(df, "CC")
    assert "p_value" in result or "error" in result


def test_mann_whitney_missing_column():
    df = _make_df()
    result = mann_whitney_test(df, "NonExistent")
    assert "error" in result


def test_mann_whitney_small_group():
    df = pd.DataFrame({
        "CC": [5, 10, 20],
        "Risk": ["Low", "High", "High"],
    })
    result = mann_whitney_test(df, "CC")
    assert "error" in result


# ── Heuristic risk ────────────────────────────────────────────────────────
def test_heuristic_risk_returns_series():
    df = _make_df()
    scores = compute_heuristic_risk(df)
    assert len(scores) == len(df)
    assert (scores.dropna() >= 0).all()
    assert (scores.dropna() <= 100).all()


def test_heuristic_risk_no_feature_cols():
    df = pd.DataFrame({"Module": ["A", "B"]})
    scores = compute_heuristic_risk(df)
    assert scores.isna().all()


# ── IQR outlier detection ────────────────────────────────────────────────
def test_flag_outliers_iqr():
    series = pd.Series([5, 6, 7, 8, 9, 100])  # 100 is an outlier
    flags = flag_outliers_iqr(series)
    assert flags.iloc[-1] == True


# ── Control chart ────────────────────────────────────────────────────────
def test_imr_chart_computed():
    series = pd.Series(range(10))
    result = compute_imr_chart(series)
    assert "cl" in result
    assert "ucl" in result
    assert "lcl" in result


def test_imr_chart_needs_4_points():
    series = pd.Series([1, 2, 3])
    result = compute_imr_chart(series)
    assert "error" in result
