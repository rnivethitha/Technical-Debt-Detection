# src/model_training.py
# Decision-tree training, evaluation, and rule extraction.
# Uses scikit-learn Pipelines to prevent data leakage.

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import warnings


def prepare_data(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "Risk",
) -> tuple | dict:
    """Extract and validate features and target.

    Returns (X, y, label_classes) or {'error': message}.
    """
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        return {"error": f"Feature columns not found: {missing_features}"}
    if target_col not in df.columns:
        return {"error": f"Target column '{target_col}' not found."}

    sub = df[feature_cols + [target_col]].copy()
    sub = sub[sub[target_col].notna()]
    sub = sub[sub[target_col].isin(["High", "Low"])]

    if len(sub) < 10:
        return {"error": f"Only {len(sub)} labelled rows available. Minimum 10 required."}

    X = sub[feature_cols]
    y = sub[target_col]

    classes = sorted(y.unique().tolist())
    if len(classes) < 2:
        return {"error": "Only one Risk class present. Supervised training requires both 'High' and 'Low'."}

    # Encode: Low=0, High=1
    y_enc = (y == "High").astype(int)

    return X, y_enc, classes


def train_decision_tree(
    X: pd.DataFrame,
    y: pd.Series,
    max_depth: int = 3,
    min_samples_leaf: int = 2,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict:
    """Train a decision-tree classifier and return complete evaluation results.

    Uses a scikit-learn Pipeline with median imputation to prevent leakage.
    """
    # Stratified split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    except ValueError:
        # Fall back to non-stratified if classes too small
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
        )

    # Pipeline: impute → decision tree
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight="balanced",
        )),
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Majority-class baseline
    majority_class = y_train.mode()[0]
    baseline_pred = np.full(len(y_test), majority_class)
    baseline_acc = accuracy_score(y_test, baseline_pred)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=["Low", "High"], output_dict=True)

    # Feature importances from the tree
    clf = pipe.named_steps["clf"]
    importances = dict(zip(X.columns, clf.feature_importances_))

    # Cross-validation (only if enough samples)
    cv_results = {}
    if len(X) >= 30:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv = cross_validate(
                pipe, X, y, cv=skf,
                scoring=["accuracy", "f1"],
                return_train_score=False,
            )
        cv_results = {
            "cv_accuracy_mean": float(cv["test_accuracy"].mean()),
            "cv_accuracy_std": float(cv["test_accuracy"].std()),
            "cv_f1_mean": float(cv["test_f1"].mean()),
            "cv_f1_std": float(cv["test_f1"].std()),
        }

    # Human-readable decision rules
    rules_text = export_text(clf, feature_names=list(X.columns))

    return {
        "pipeline": pipe,
        "clf": clf,
        "feature_names": list(X.columns),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "baseline_accuracy": float(baseline_acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": cr,
        "feature_importances": importances,
        "rules_text": rules_text,
        "cv_results": cv_results,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "max_depth": max_depth,
    }


def plot_decision_tree(clf, feature_names: list, class_names=("Low", "High")) -> bytes:
    """Return PNG bytes of the decision-tree diagram."""
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=list(class_names),
        filled=True,
        rounded=True,
        ax=ax,
        fontsize=9,
        impurity=False,
        proportion=False,
    )
    ax.set_title("Decision Tree — Technical Debt Risk Classifier", fontsize=12, fontweight="bold")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def plot_feature_importance(importances: dict) -> bytes:
    """Return PNG bytes of a horizontal bar chart of feature importances."""
    if not importances:
        return b""
    sorted_items = sorted(importances.items(), key=lambda x: x[1])
    features, values = zip(*sorted_items)
    colors = ["#E05252" if v > 0.2 else "#5B8DB8" for v in values]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(features, values, color=colors, edgecolor="white")
    ax.set_xlabel("Feature Importance (Gini)", fontsize=10)
    ax.set_title("Feature Importance", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9,
        )
    ax.set_xlim(0, max(values) * 1.3 + 0.01)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def plot_confusion_matrix(cm: list, class_names=("Low", "High")) -> bytes:
    """Return PNG bytes of the confusion matrix heatmap."""
    import seaborn as sns
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def predict_module(pipeline, feature_names: list, row: pd.Series) -> dict:
    """Predict risk for a single module row using the trained pipeline.

    Returns predicted label (High/Low), probability, and applicable rule.
    """
    X_row = row[feature_names].to_frame().T.reset_index(drop=True)
    pred_int = pipeline.predict(X_row)[0]
    pred_label = "High" if pred_int == 1 else "Low"

    try:
        proba = pipeline.predict_proba(X_row)[0]
        confidence = float(max(proba)) * 100
    except Exception:
        confidence = None

    return {
        "predicted_label": pred_label,
        "confidence_pct": confidence,
    }
