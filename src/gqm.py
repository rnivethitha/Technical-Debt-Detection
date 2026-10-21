# src/gqm.py
# GQM (Goal-Question-Metric) framework data and computations.
# Maps project goals → questions → metrics → computed values.

import pandas as pd
import numpy as np
from src.metrics_analysis import compute_descriptive_stats, flag_outliers_iqr


def gqm_g1_complexity(df: pd.DataFrame) -> dict:
    """G1: Identify modules with elevated technical debt risk.

    Q1: Which modules have unusually high complexity?
    Q2: Do high-risk and low-risk modules differ significantly?
    Q3: Which metrics move together?
    """
    results = {}

    # Q1 – CC distribution
    if "CC" in df.columns:
        cc = df["CC"].dropna()
        q1, med, q3 = cc.quantile(0.25), cc.median(), cc.quantile(0.75)
        iqr = q3 - q1
        outliers = df[flag_outliers_iqr(cc)]["Module"].tolist() if "Module" in df.columns else []
        results["Q1"] = {
            "metric": "Cyclomatic Complexity (CC)",
            "count": int(cc.count()),
            "mean": round(float(cc.mean()), 2),
            "median": round(float(med), 2),
            "q1": round(float(q1), 2),
            "q3": round(float(q3), 2),
            "iqr": round(float(iqr), 2),
            "outlier_threshold": round(float(q3 + 1.5 * iqr), 2),
            "outlier_modules": outliers,
            "n_outliers": len(outliers),
        }
    else:
        results["Q1"] = {"metric": "CC", "error": "CC column not available."}

    # Q2 – Group comparison
    if "Risk" in df.columns:
        metrics_to_compare = [c for c in ["LOC", "CC", "Dup", "Churn", "Coupling"] if c in df.columns]
        group_stats = {}
        for col in metrics_to_compare:
            high = df[df["Risk"] == "High"][col].dropna()
            low = df[df["Risk"] == "Low"][col].dropna()
            group_stats[col] = {
                "high_mean": round(float(high.mean()), 2) if len(high) else "N/A",
                "low_mean": round(float(low.mean()), 2) if len(low) else "N/A",
                "high_median": round(float(high.median()), 2) if len(high) else "N/A",
                "low_median": round(float(low.median()), 2) if len(low) else "N/A",
            }
        results["Q2"] = {"group_stats": group_stats}
    else:
        results["Q2"] = {"error": "No 'Risk' column available for group comparison."}

    return results


def gqm_g2_monitoring() -> dict:
    """G2: Monitor changes across builds.

    Returns instructions for the control chart (build data needed).
    """
    return {
        "description": (
            "G2 requires ordered build-level data with a build identifier column. "
            "If your dataset has such a column, enable the control chart on the "
            "'Statistical Visualisation' page."
        ),
        "metrics": ["CC per build", "Centre line", "UCL", "LCL", "Out-of-control signals"],
    }


def gqm_g3_model(train_result: dict | None) -> dict:
    """G3: Evaluate the decision tree.

    Q1: Accuracy, precision, recall, F1, confusion matrix vs baseline.
    Q2: Tree depth, feature importance, human-readable rules.
    """
    if train_result is None:
        return {"error": "No trained model available. Train the model on the Decision Tree page."}

    return {
        "Q1": {
            "accuracy": train_result.get("accuracy"),
            "precision": train_result.get("precision"),
            "recall": train_result.get("recall"),
            "f1": train_result.get("f1"),
            "baseline_accuracy": train_result.get("baseline_accuracy"),
            "n_train": train_result.get("n_train"),
            "n_test": train_result.get("n_test"),
            "cv_results": train_result.get("cv_results", {}),
        },
        "Q2": {
            "max_depth": train_result.get("max_depth"),
            "feature_importances": train_result.get("feature_importances"),
            "rules": train_result.get("rules_text"),
        },
    }


def gqm_g4_usability() -> dict:
    """G4: Evaluate usability (task-based + optional SUS).
    Data is recorded on the Usability page.
    """
    return {
        "tasks": [
            "Upload a valid metrics CSV",
            "Find data-quality warnings",
            "Read a box plot",
            "Identify a point outside control limits",
            "Find module predictions and explain a decision rule",
            "Export a report",
        ],
        "metrics": [
            "Task completion rate (%)",
            "Time per task (seconds)",
            "Error count",
            "System Usability Scale (SUS) score (optional)",
        ],
    }


def gqm_g5_function_points(fp_result: dict | None) -> dict:
    """G5: Estimate functional size using IFPUG Function Points."""
    if fp_result is None:
        return {"error": "No Function Point calculation performed yet."}
    return fp_result
