# tests/test_validation.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np
from src.validation import validate_dataframe, clean_dataframe


def _make_df(**overrides):
    """Helper: create a minimal valid dataframe."""
    data = {
        "Module": ["A", "B", "C", "D", "E"],
        "LOC":    [100, 200, 300, 150, 500],
        "CC":     [5, 8, 12, 7, 20],
        "Dup":    [5.0, 10.0, 15.0, 8.0, 25.0],
        "Churn":  [10, 20, 30, 15, 40],
        "Coupling": [3, 5, 8, 4, 12],
        "Risk":   ["Low", "Low", "High", "Low", "High"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


# ── Valid dataset ─────────────────────────────────────────────────────────
def test_valid_dataset_no_errors():
    df = _make_df()
    result = validate_dataframe(df)
    assert result["errors"] == [], f"Unexpected errors: {result['errors']}"


def test_valid_dataset_row_count():
    df = _make_df()
    result = validate_dataframe(df)
    assert result["summary"]["row_count"] == 5


# ── Empty dataset ─────────────────────────────────────────────────────────
def test_empty_dataframe_raises_error():
    df = pd.DataFrame()
    result = validate_dataframe(df)
    assert any("zero rows" in e.lower() for e in result["errors"])


# ── Missing required columns ─────────────────────────────────────────────
def test_missing_required_column_produces_warning():
    df = _make_df()
    df = df.drop(columns=["CC"])
    result = validate_dataframe(df)
    assert any("CC" in w for w in result["warnings"])


# ── Missing values ────────────────────────────────────────────────────────
def test_missing_values_detected():
    df = _make_df()
    df.loc[0, "LOC"] = np.nan
    result = validate_dataframe(df)
    assert result["summary"]["total_missing"] == 1


# ── Duplicate rows ────────────────────────────────────────────────────────
def test_duplicate_rows_detected():
    df = _make_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    result = validate_dataframe(df)
    assert result["summary"]["duplicate_rows"] >= 1


# ── Duplicate module IDs ──────────────────────────────────────────────────
def test_duplicate_module_id_detected():
    df = _make_df(Module=["A", "A", "C", "D", "E"])
    result = validate_dataframe(df)
    assert result["summary"]["duplicate_module_ids"] >= 1


# ── Invalid labels ────────────────────────────────────────────────────────
def test_invalid_risk_labels_produce_warning():
    df = _make_df(Risk=["Low", "Medium", "High", "Unknown", "Low"])
    result = validate_dataframe(df)
    assert any("unexpected" in w.lower() or "invalid" in w.lower() for w in result["warnings"])


# ── Single class ──────────────────────────────────────────────────────────
def test_single_class_target_produces_error():
    df = _make_df(Risk=["Low", "Low", "Low", "Low", "Low"])
    result = validate_dataframe(df)
    assert any("one Risk class" in e or "one class" in e.lower() for e in result["errors"])


# ── Negative values ───────────────────────────────────────────────────────
def test_negative_values_produce_error():
    df = _make_df(LOC=[100, -50, 300, 150, 500])
    result = validate_dataframe(df)
    assert any("negative" in e.lower() for e in result["errors"])


# ── Small dataset warning ─────────────────────────────────────────────────
def test_small_dataset_warning():
    df = _make_df()  # 5 rows
    result = validate_dataframe(df)
    assert any("only" in w.lower() or "minimum" in w.lower() for w in result["warnings"])


# ── Clean dataframe ───────────────────────────────────────────────────────
def test_clean_removes_duplicates():
    df = _make_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    df["_is_demo"] = False
    cleaned, actions = clean_dataframe(df)
    assert len(cleaned) == 5
    assert any("duplicate" in a.lower() for a in actions)


def test_clean_normalises_risk_labels():
    df = _make_df(Risk=["low", "HIGH", "High", "Low", "LOW"])
    df["_is_demo"] = False
    cleaned, _ = clean_dataframe(df)
    assert set(cleaned["Risk"].dropna().unique()).issubset({"High", "Low"})
