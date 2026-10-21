# src/data_loader.py
# Handles loading and initial processing of CSV files.
# Supports both the built-in demo dataset and user-uploaded files.

import pandas as pd
import os


DEMO_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "demo_modules.csv")

# These are the standard column names used internally throughout the app.
REQUIRED_COLUMNS = ["LOC", "CC", "Dup", "Churn", "Coupling"]
OPTIONAL_COLUMNS = ["Maintainability", "Risk"]
MODULE_ID_COLUMN = "Module"


def load_demo_dataset() -> pd.DataFrame:
    """Load the built-in demonstration dataset.

    Returns a DataFrame clearly marked as demo data.
    The Risk column contains pre-assigned illustrative labels.
    """
    path = os.path.abspath(DEMO_PATH)
    df = pd.read_csv(path)
    df["_is_demo"] = True  # internal flag
    return df


def load_uploaded_csv(file_obj) -> pd.DataFrame:
    """Load a user-supplied CSV file object (from st.file_uploader).

    Returns a DataFrame without any internal demo flag.
    Raises ValueError for empty or unreadable files.
    """
    try:
        df = pd.read_csv(file_obj)
    except Exception as exc:
        raise ValueError(f"Could not parse CSV file: {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded file is empty.")

    df["_is_demo"] = False
    return df


def apply_column_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Rename columns in df according to the user-supplied mapping.

    mapping example:
        {
            "Module": "FileName",
            "LOC":    "lines_of_code",
            "CC":     "cyclomatic_complexity",
            ...
        }
    Only keys present in the mapping AND present as values in the df are renamed.
    """
    reverse = {v: k for k, v in mapping.items() if v and v in df.columns}
    return df.rename(columns=reverse)


def get_numeric_columns(df: pd.DataFrame) -> list:
    """Return list of numeric column names, excluding internal metadata columns."""
    exclude = {"_is_demo"}
    return [c for c in df.select_dtypes(include="number").columns if c not in exclude]


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the subset of standard feature columns that are present in df."""
    return [c for c in REQUIRED_COLUMNS if c in df.columns]
