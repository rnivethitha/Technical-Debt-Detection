# tests/test_model.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np
from src.model_training import prepare_data, train_decision_tree, predict_module


def _make_labelled_df(n=40, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Module": [f"M{i}" for i in range(n)],
        "LOC":    list(rng.integers(100, 800, n)),
        "CC":     list(rng.integers(3, 25, n)),
        "Dup":    list(rng.uniform(2, 35, n).round(1)),
        "Churn":  list(rng.integers(5, 70, n)),
        "Coupling": list(rng.integers(2, 16, n)),
        "Risk":   (["High"] * (n // 2)) + (["Low"] * (n // 2)),
    })


# ── prepare_data ──────────────────────────────────────────────────────────
def test_prepare_data_success():
    df = _make_labelled_df()
    result = prepare_data(df, ["LOC", "CC", "Dup", "Churn", "Coupling"])
    assert not isinstance(result, dict), "Should not return error dict"
    X, y, classes = result
    assert len(X) == len(y)
    assert set(classes) == {"High", "Low"}


def test_prepare_data_missing_feature_col():
    df = _make_labelled_df()
    result = prepare_data(df, ["LOC", "NonExistentCol"])
    assert isinstance(result, dict) and "error" in result


def test_prepare_data_missing_target():
    df = _make_labelled_df()
    df = df.drop(columns=["Risk"])
    result = prepare_data(df, ["LOC", "CC"])
    assert isinstance(result, dict) and "error" in result


def test_prepare_data_single_class():
    df = _make_labelled_df()
    df["Risk"] = "Low"
    result = prepare_data(df, ["LOC", "CC"])
    assert isinstance(result, dict) and "error" in result


# ── train_decision_tree ───────────────────────────────────────────────────
def test_train_returns_expected_keys():
    df = _make_labelled_df()
    X, y, _ = prepare_data(df, ["LOC", "CC", "Dup", "Churn", "Coupling"])
    result = train_decision_tree(X, y, max_depth=3, random_state=42)
    assert "accuracy" in result
    assert "f1" in result
    assert "confusion_matrix" in result
    assert "feature_importances" in result
    assert "rules_text" in result


def test_train_accuracy_is_float():
    df = _make_labelled_df()
    X, y, _ = prepare_data(df, ["LOC", "CC"])
    result = train_decision_tree(X, y, random_state=42)
    assert isinstance(result["accuracy"], float)
    assert 0.0 <= result["accuracy"] <= 1.0


def test_train_baseline_accuracy():
    df = _make_labelled_df(n=40)
    X, y, _ = prepare_data(df, ["LOC", "CC"])
    result = train_decision_tree(X, y, random_state=42)
    assert 0.0 <= result["baseline_accuracy"] <= 1.0


def test_train_confusion_matrix_shape():
    df = _make_labelled_df()
    X, y, _ = prepare_data(df, ["LOC", "CC"])
    result = train_decision_tree(X, y, random_state=42)
    cm = np.array(result["confusion_matrix"])
    assert cm.shape == (2, 2)


def test_train_feature_importances_sum_to_one():
    df = _make_labelled_df()
    feats = ["LOC", "CC", "Dup"]
    X, y, _ = prepare_data(df, feats)
    result = train_decision_tree(X, y, random_state=42)
    total = sum(result["feature_importances"].values())
    assert abs(total - 1.0) < 1e-6


def test_reproducibility():
    df = _make_labelled_df()
    X, y, _ = prepare_data(df, ["LOC", "CC", "Dup"])
    r1 = train_decision_tree(X, y, random_state=42)
    r2 = train_decision_tree(X, y, random_state=42)
    assert r1["accuracy"] == r2["accuracy"]


# ── predict_module ────────────────────────────────────────────────────────
def test_predict_module_returns_label():
    df = _make_labelled_df()
    feats = ["LOC", "CC", "Dup", "Churn", "Coupling"]
    X, y, _ = prepare_data(df, feats)
    result = train_decision_tree(X, y, random_state=42)
    row = df.iloc[0]
    pred = predict_module(result["pipeline"], feats, row)
    assert pred["predicted_label"] in ["High", "Low"]


def test_predict_module_confidence_in_range():
    df = _make_labelled_df()
    feats = ["LOC", "CC"]
    X, y, _ = prepare_data(df, feats)
    result = train_decision_tree(X, y, random_state=42)
    row = df.iloc[5]
    pred = predict_module(result["pipeline"], feats, row)
    if pred["confidence_pct"] is not None:
        assert 0.0 <= pred["confidence_pct"] <= 100.0
