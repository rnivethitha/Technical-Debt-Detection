# src/validation.py
# Validates uploaded datasets and returns structured quality reports.

import pandas as pd
import numpy as np
from src.data_loader import REQUIRED_COLUMNS, MODULE_ID_COLUMN


def validate_dataframe(df: pd.DataFrame) -> dict:
    """Run all validation checks on the dataframe.

    Returns a dict with:
        - 'errors':   list of blocking issues (prevent analysis)
        - 'warnings': list of non-blocking notices
        - 'summary':  dict of quality metrics
    """
    errors = []
    warnings = []
    summary = {}

    # ── Basic shape ──────────────────────────────────────────────────────────
    summary["row_count"] = len(df)
    summary["col_count"] = len(df.columns)

    if len(df) == 0:
        errors.append("Dataset contains zero rows.")
        return {"errors": errors, "warnings": warnings, "summary": summary}

    # ── Required metric columns ───────────────────────────────────────────────
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        warnings.append(
            f"Standard metric columns not found: {missing_cols}. "
            "Use the column-mapping tool on the Upload page to map your columns."
        )

    # ── Module identifier ────────────────────────────────────────────────────
    if MODULE_ID_COLUMN not in df.columns:
        warnings.append(
            f"No '{MODULE_ID_COLUMN}' column detected. "
            "Module-level predictions will use row numbers."
        )

    # ── Missing values ───────────────────────────────────────────────────────
    missing_counts = df.isnull().sum()
    total_missing = int(missing_counts.sum())
    summary["total_missing"] = total_missing
    summary["missing_by_column"] = missing_counts[missing_counts > 0].to_dict()

    if total_missing > 0:
        cols_with_na = [c for c, n in summary["missing_by_column"].items()]
        warnings.append(
            f"Missing values detected in {len(cols_with_na)} column(s): "
            f"{cols_with_na}. "
            "Missing values will be imputed using column medians during ML training."
        )

    # ── Duplicate rows ────────────────────────────────────────────────────────
    n_dup_rows = int(df.duplicated().sum())
    summary["duplicate_rows"] = n_dup_rows
    if n_dup_rows > 0:
        warnings.append(f"{n_dup_rows} fully duplicate row(s) detected.")

    # ── Duplicate module identifiers ─────────────────────────────────────────
    if MODULE_ID_COLUMN in df.columns:
        n_dup_id = int(df[MODULE_ID_COLUMN].duplicated().sum())
        summary["duplicate_module_ids"] = n_dup_id
        if n_dup_id > 0:
            warnings.append(f"{n_dup_id} duplicate module identifier(s) detected.")
    else:
        summary["duplicate_module_ids"] = "N/A"

    # ── Numeric columns: negative values ────────────────────────────────────
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols:
        if col.startswith("_"):
            continue
        if col in REQUIRED_COLUMNS or col in ["Maintainability"]:
            if (df[col] < 0).any():
                errors.append(
                    f"Column '{col}' contains negative values, "
                    "which are invalid for this metric."
                )

    # ── Data types ──────────────────────────────────────────────────────────
    summary["dtypes"] = {c: str(df[c].dtype) for c in df.columns if not c.startswith("_")}

    # ── Target label ('Risk') ────────────────────────────────────────────────
    risk_summary = {}
    if "Risk" in df.columns:
        valid_labels = {"Low", "High"}
        raw_labels = df["Risk"].dropna().unique().tolist()
        # Normalise
        normed = df["Risk"].astype(str).str.strip().str.capitalize()
        invalid_labels = [v for v in normed.unique() if v not in valid_labels and v != "Nan"]
        if invalid_labels:
            warnings.append(
                f"Unexpected Risk labels: {invalid_labels}. "
                "Expected 'High' or 'Low'. Non-standard values will be treated as missing."
            )
        class_counts = normed.value_counts().to_dict()
        risk_summary["class_distribution"] = class_counts
        n_classes = len([k for k in class_counts if k in valid_labels])
        risk_summary["n_classes"] = n_classes
        if n_classes == 1:
            errors.append(
                "Only one Risk class present in the dataset. "
                "Supervised model training requires at least two classes (High and Low)."
            )
        elif n_classes == 0:
            warnings.append("No valid Risk labels found. Supervised training is not possible.")
    else:
        risk_summary["class_distribution"] = {}
        risk_summary["n_classes"] = 0
        warnings.append(
            "No 'Risk' column found. Heuristic risk scoring will be used instead of supervised ML."
        )

    summary["risk"] = risk_summary

    # ── Small dataset warning ────────────────────────────────────────────────
    if len(df) < 20:
        warnings.append(
            f"Dataset has only {len(df)} rows. "
            "Results may be unreliable; a minimum of 20–30 rows is recommended."
        )

    return {"errors": errors, "warnings": warnings, "summary": summary}


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Apply safe, documented cleaning steps. Returns (cleaned_df, list_of_actions)."""
    actions = []
    df = df.copy()

    # Drop fully duplicate rows
    n_before = len(df)
    df = df.drop_duplicates()
    n_dropped = n_before - len(df)
    if n_dropped:
        actions.append(f"Removed {n_dropped} fully duplicate row(s).")

    # Normalise Risk column if present
    if "Risk" in df.columns:
        df["Risk"] = df["Risk"].astype(str).str.strip().str.capitalize()
        df.loc[~df["Risk"].isin(["High", "Low"]), "Risk"] = np.nan
        n_bad = df["Risk"].isna().sum()
        if n_bad:
            actions.append(
                f"Set {n_bad} invalid Risk label(s) to NaN "
                "(will be excluded from supervised training)."
            )

    # Clip negative metric values to zero with a warning (only for non-demo)
    from src.data_loader import REQUIRED_COLUMNS
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            n_neg = (df[col] < 0).sum()
            if n_neg:
                df[col] = df[col].clip(lower=0)
                actions.append(
                    f"Clipped {n_neg} negative value(s) in '{col}' to 0. "
                    "Review your data source."
                )

    return df, actions
