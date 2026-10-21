# app.py  — Main Streamlit application
# AI-Based Technical Debt Detection Using Software Metrics
# Run: python -m streamlit run app.py

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import io
import datetime

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="AI-Based Technical Debt Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Source modules ────────────────────────────────────────────────────────────
from src.data_loader import (
    load_demo_dataset, load_uploaded_csv, apply_column_mapping,
    get_feature_columns, REQUIRED_COLUMNS, MODULE_ID_COLUMN,
)
from src.validation import validate_dataframe, clean_dataframe
from src.metrics_analysis import (
    compute_descriptive_stats, compute_heuristic_risk,
    flag_outliers_iqr, METRIC_DEFINITIONS,
)
from src.statistics_analysis import (
    pearson_correlation, spearman_correlation, mann_whitney_test,
    plot_correlation_heatmap, plot_boxplot,
)
from src.control_chart import plot_imr_chart, compute_imr_chart
from src.model_training import (
    prepare_data, train_decision_tree,
    plot_decision_tree, plot_feature_importance, plot_confusion_matrix,
    predict_module,
)
from src.function_points import (
    calculate_ufp, calculate_vaf, calculate_afp, estimate_effort,
    get_demo_function_inventory, get_demo_gsc_ratings,
    WEIGHTS, GSC_NAMES,
)
from src.gqm import gqm_g1_complexity, gqm_g2_monitoring, gqm_g3_model, gqm_g4_usability
from src.report_generator import build_pdf_report
from src.usability import TASKS, SUS_QUESTIONS, calculate_sus, calculate_completion_rate

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a2744 0%, #0f1c38 100%);
    border-right: 1px solid #2a3f6f;
}
[data-testid="stSidebar"] * { color: #d0daf0 !important; }
[data-testid="stSidebar"] .stRadio label { color: #a8c0e8 !important; font-size: 0.93rem; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e3a5f 0%, #2a4f7c 100%);
    border-radius: 12px; padding: 18px 22px;
    color: white; margin-bottom: 10px;
    border-left: 4px solid #2980b9;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.metric-card .value { font-size: 2rem; font-weight: 700; color: #5dade2; }
.metric-card .label { font-size: 0.8rem; color: #a8c0e8; text-transform: uppercase; letter-spacing: 0.08em; }

/* Demo banner */
.demo-banner {
    background: linear-gradient(90deg, #7d3c08, #a04000);
    color: #ffe0b2; border-radius: 8px; padding: 10px 16px;
    font-weight: 600; margin-bottom: 14px;
    border-left: 4px solid #ff9800;
}

/* Section headers */
h1 { color: #1a2744 !important; }
h2 { color: #2a4f7c !important; }
h3 { color: #2980b9 !important; }

/* Success / error / warning */
.stAlert { border-radius: 8px; }

/* Tables */
.stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session state initialisation ─────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "is_demo" not in st.session_state:
    st.session_state.is_demo = False
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = ""
if "validation" not in st.session_state:
    st.session_state.validation = None
if "train_result" not in st.session_state:
    st.session_state.train_result = None
if "fp_result" not in st.session_state:
    st.session_state.fp_result = None
if "predictions_df" not in st.session_state:
    st.session_state.predictions_df = None
if "tree_png" not in st.session_state:
    st.session_state.tree_png = None
if "corr_png" not in st.session_state:
    st.session_state.corr_png = None
if "box_png" not in st.session_state:
    st.session_state.box_png = None


def demo_banner():
    if st.session_state.is_demo:
        st.markdown(
            '<div class="demo-banner">⚠️ DEMO DATA: Results are illustrative and must not be '
            'presented as real-world experimental findings.</div>',
            unsafe_allow_html=True,
        )


def metric_card(label, value, col=None):
    html = f'<div class="metric-card"><div class="value">{value}</div><div class="label">{label}</div></div>'
    if col:
        col.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔍 Technical Debt Detection")
    st.markdown("**Software Metrics Analysis,<br>Risk Prediction & Decision Support**", unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "📤 Dataset Upload",
            "✅ Data Quality",
            "📐 GQM & Metric Analysis",
            "📊 Statistical Visualisation",
            "🌳 Decision Tree & Model",
            "🎯 Risk Prediction",
            "📄 Report & Export",
            "🖩 Function Point Analysis",
            "👥 About / Usability",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    if st.session_state.df is not None:
        n = len(st.session_state.df)
        label = "🟡 Demo" if st.session_state.is_demo else "🟢 Uploaded"
        st.markdown(f"**Dataset:** {st.session_state.dataset_name}")
        st.markdown(f"**Mode:** {label}")
        st.markdown(f"**Rows:** {n}")
    else:
        st.markdown("*No dataset loaded.*")
        st.markdown("Go to **Dataset Upload** to begin.")

    st.markdown("---")
    st.markdown("<small>AI-Based Technical Debt Detection v1.0<br>© 2024 — College Project Demo</small>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🔍 AI-Based Technical Debt Detection")
    st.markdown("### Software Metrics Analysis, Risk Prediction and Decision Support")
    st.markdown("---")

    demo_banner()

    if st.session_state.df is None:
        st.info("👈 No dataset loaded. Click **Dataset Upload** in the sidebar to begin.")
        st.markdown("""
        #### What is Technical Debt?
        Technical debt is the extra development work that arises when code that is easy to implement
        in the short run is used instead of applying the best overall solution.

        #### What does this tool do?
        This application analyses software module metrics and flags modules that may have elevated
        technical debt risk, using a transparent **Decision Tree classifier**.

        #### Quick Start
        1. Go to **Dataset Upload** and choose **Demo Dataset** or upload your own CSV.
        2. Check **Data Quality** for any issues.
        3. Explore **Statistical Visualisation** for charts.
        4. Train the model on **Decision Tree & Model**.
        5. View per-module predictions on **Risk Prediction**.
        6. Download a PDF on **Report & Export**.
        """)
    else:
        df = st.session_state.df
        val = st.session_state.validation or {}
        summary = val.get("summary", {})
        risk_info = summary.get("risk", {})
        class_dist = risk_info.get("class_distribution", {})

        high_n = class_dist.get("High", 0)
        low_n = class_dist.get("Low", 0)
        total = summary.get("row_count", len(df))
        high_pct = f"{100*high_n/total:.1f}%" if total else "N/A"
        low_pct = f"{100*low_n/total:.1f}%" if total else "N/A"

        feature_cols = get_feature_columns(df)
        mean_loc = f"{df['LOC'].mean():.0f}" if "LOC" in df.columns else "N/A"
        mean_cc = f"{df['CC'].mean():.1f}" if "CC" in df.columns else "N/A"

        col1, col2, col3, col4 = st.columns(4)
        metric_card("Modules Analysed", total, col1)
        metric_card("High-Risk Modules", f"{high_n} ({high_pct})", col2)
        metric_card("Low-Risk Modules", f"{low_n} ({low_pct})", col3)
        metric_card("Missing Values", summary.get("total_missing", "N/A"), col4)

        col5, col6, col7, col8 = st.columns(4)
        metric_card("Duplicate Rows", summary.get("duplicate_rows", "N/A"), col5)
        metric_card("Mean LOC", mean_loc, col6)
        metric_card("Mean CC", mean_cc, col7)
        metric_card("Model Trained", "Yes ✅" if st.session_state.train_result else "No ❌", col8)

        st.markdown("---")
        st.markdown("#### ℹ️ What does 'Technical Debt Risk' mean?")
        st.info(
            "**High Risk** indicates that a module has metric values associated with elevated "
            "maintenance cost, complexity or instability. It is an **indicator for human review**, "
            "not proof that the module is defective. Always validate findings with code review."
        )

        st.markdown("#### 📋 Dataset Preview")
        st.dataframe(df.drop(columns=["_is_demo"], errors="ignore").head(10), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATASET UPLOAD
# ════════════════════════════════════════════════════════════════════════════
elif page == "📤 Dataset Upload":
    st.title("📤 Dataset Upload")
    st.markdown("Choose the **Demo Dataset** to explore functionality, or upload your own CSV.")
    st.markdown("---")

    tab_demo, tab_upload = st.tabs(["🧪 Mode A — Demo Dataset", "📁 Mode B — Upload CSV"])

    with tab_demo:
        st.markdown("#### Demonstration Dataset (30 synthetic software modules)")
        st.warning(
            "⚠️ DEMO DATA: This dataset is synthetic. Results are illustrative only and "
            "must NOT be presented as real-world experimental findings."
        )
        st.markdown("""
        **Schema:** `Module, LOC, CC, Dup, Churn, Coupling, Risk`  
        **Rows:** 30 modules  
        **Labels:** High / Low (illustrative)  
        """)
        if st.button("Load Demo Dataset", type="primary"):
            with st.spinner("Loading demo dataset..."):
                df = load_demo_dataset()
                val = validate_dataframe(df)
                df, actions = clean_dataframe(df)
                st.session_state.df = df
                st.session_state.is_demo = True
                st.session_state.dataset_name = "Demo Dataset (Synthetic)"
                st.session_state.validation = val
                st.session_state.train_result = None
                st.session_state.predictions_df = None
            st.success(f"✅ Demo dataset loaded — {len(df)} modules.")
            st.dataframe(df.drop(columns=["_is_demo"], errors="ignore"), use_container_width=True)

    with tab_upload:
        st.markdown("#### Upload Your Own CSV Metrics File")
        uploaded = st.file_uploader("Select a CSV file", type=["csv"])

        if uploaded:
            try:
                raw_df = load_uploaded_csv(uploaded)
                st.success(f"File read: {len(raw_df)} rows, {len(raw_df.columns)} columns.")
                st.markdown("**Preview (first 5 rows):**")
                st.dataframe(raw_df.head(5), use_container_width=True)

                # Column mapping
                st.markdown("#### Column Mapping")
                st.info(
                    "Map your CSV columns to the standard metric names. "
                    "Leave blank if a column is not present in your file."
                )
                all_cols = ["(none)"] + list(raw_df.columns)
                mapping = {}
                col1, col2 = st.columns(2)
                with col1:
                    mapping["Module"] = st.selectbox("Module identifier", all_cols, index=0)
                    mapping["LOC"] = st.selectbox("Lines of Code (LOC)", all_cols, index=0)
                    mapping["CC"] = st.selectbox("Cyclomatic Complexity (CC)", all_cols, index=0)
                    mapping["Dup"] = st.selectbox("Code Duplication % (Dup)", all_cols, index=0)
                with col2:
                    mapping["Churn"] = st.selectbox("Code Churn", all_cols, index=0)
                    mapping["Coupling"] = st.selectbox("Coupling", all_cols, index=0)
                    mapping["Maintainability"] = st.selectbox("Maintainability Index (optional)", all_cols, index=0)
                    mapping["Risk"] = st.selectbox("Risk Label (optional)", all_cols, index=0)

                # Build mapping (filter out "(none)")
                clean_mapping = {k: v for k, v in mapping.items() if v != "(none)"}

                if st.button("Apply Mapping & Validate", type="primary"):
                    df = apply_column_mapping(raw_df, clean_mapping)
                    val = validate_dataframe(df)
                    df, actions = clean_dataframe(df)

                    st.session_state.df = df
                    st.session_state.is_demo = False
                    st.session_state.dataset_name = uploaded.name
                    st.session_state.validation = val
                    st.session_state.train_result = None
                    st.session_state.predictions_df = None

                    if val["errors"]:
                        for e in val["errors"]:
                            st.error(f"❌ {e}")
                    if val["warnings"]:
                        for w in val["warnings"]:
                            st.warning(f"⚠️ {w}")
                    if actions:
                        st.info("**Cleaning actions performed:**\n" + "\n".join(f"• {a}" for a in actions))
                    if not val["errors"]:
                        st.success(f"✅ Dataset validated — {len(df)} usable rows.")

            except ValueError as ex:
                st.error(f"❌ {ex}")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATA QUALITY
# ════════════════════════════════════════════════════════════════════════════
elif page == "✅ Data Quality":
    st.title("✅ Data Quality Report")
    demo_banner()

    if st.session_state.df is None:
        st.warning("No dataset loaded. Go to **Dataset Upload** first.")
        st.stop()

    df = st.session_state.df
    val = st.session_state.validation or {}
    summary = val.get("summary", {})

    # Errors / Warnings
    errors = val.get("errors", [])
    warnings = val.get("warnings", [])

    if errors:
        st.markdown("#### ❌ Blocking Errors")
        for e in errors:
            st.error(e)
    else:
        st.success("✅ No blocking errors detected.")

    if warnings:
        st.markdown("#### ⚠️ Warnings")
        for w in warnings:
            st.warning(w)

    st.markdown("---")
    st.markdown("#### 📋 Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    metric_card("Total Rows", summary.get("row_count", "N/A"), col1)
    metric_card("Total Columns", summary.get("col_count", "N/A"), col2)
    metric_card("Missing Values", summary.get("total_missing", "N/A"), col3)
    metric_card("Duplicate Rows", summary.get("duplicate_rows", "N/A"), col4)

    # Class distribution
    risk_info = summary.get("risk", {})
    class_dist = risk_info.get("class_distribution", {})
    if class_dist:
        st.markdown("#### Class Distribution (Risk Label)")
        cd_df = pd.DataFrame(list(class_dist.items()), columns=["Risk", "Count"])
        cd_df["Percentage"] = (cd_df["Count"] / cd_df["Count"].sum() * 100).round(1).astype(str) + "%"
        st.dataframe(cd_df, use_container_width=True, hide_index=True)

    # Missing values detail
    mv = summary.get("missing_by_column", {})
    if mv:
        st.markdown("#### Missing Values by Column")
        mv_df = pd.DataFrame(list(mv.items()), columns=["Column", "Missing Count"])
        mv_df["% Missing"] = (mv_df["Missing Count"] / summary["row_count"] * 100).round(1).astype(str) + "%"
        st.dataframe(mv_df, use_container_width=True, hide_index=True)

    # Data types
    dtypes = summary.get("dtypes", {})
    if dtypes:
        st.markdown("#### Detected Data Types")
        dt_df = pd.DataFrame(list(dtypes.items()), columns=["Column", "Dtype"])
        st.dataframe(dt_df, use_container_width=True, hide_index=True)

    # Download clean CSV
    st.markdown("---")
    clean_csv = df.drop(columns=["_is_demo"], errors="ignore").to_csv(index=False)
    st.download_button(
        "⬇️ Download Validated/Cleaned CSV",
        data=clean_csv,
        file_name="cleaned_dataset.csv",
        mime="text/csv",
    )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — GQM & METRIC ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == "📐 GQM & Metric Analysis":
    st.title("📐 GQM & Metric Analysis")
    st.markdown("**Goal–Question–Metric (GQM) framework** maps project goals to measurable metrics.")
    demo_banner()

    if st.session_state.df is None:
        st.warning("No dataset loaded.")
        st.stop()

    df = st.session_state.df

    # ── Metric Definitions ────────────────────────────────────────────────
    st.markdown("### Metric Definitions")
    with st.expander("📖 Click to view all metric definitions"):
        for key, info in METRIC_DEFINITIONS.items():
            if key in df.columns or key in REQUIRED_COLUMNS:
                st.markdown(f"**{key} — {info['name']}**")
                st.markdown(f"- **Unit:** {info['unit']}")
                st.markdown(f"- **Definition:** {info['description']}")
                st.markdown(f"- **Interpretation:** {info['interpretation']}")
                st.markdown("---")

    # ── G1 ───────────────────────────────────────────────────────────────
    st.markdown("### G1 — Identify Elevated Debt Risk")
    g1 = gqm_g1_complexity(df)

    with st.expander("Q1: Which modules have unusually high complexity? (CC Distribution)", expanded=True):
        q1 = g1.get("Q1", {})
        if "error" not in q1:
            col1, col2, col3, col4 = st.columns(4)
            metric_card("Mean CC", q1["mean"], col1)
            metric_card("Median CC", q1["median"], col2)
            metric_card("IQR", q1["iqr"], col3)
            metric_card("Outlier Threshold", q1["outlier_threshold"], col4)
            if q1["outlier_modules"]:
                st.warning(f"🔴 {q1['n_outliers']} outlier module(s): {', '.join(q1['outlier_modules'])}")
                st.info("**IQR Rule:** A value is an outlier if it exceeds Q3 + 1.5 × IQR.")
            else:
                st.success("No CC outliers detected using 1.5 × IQR rule.")
        else:
            st.error(q1["error"])

    with st.expander("Q2: Do High-Risk and Low-Risk modules differ?"):
        q2 = g1.get("Q2", {})
        if "error" not in q2:
            group_df = pd.DataFrame(q2["group_stats"]).T.reset_index()
            group_df.columns = ["Metric", "High Mean", "Low Mean", "High Median", "Low Median"]
            st.dataframe(group_df, use_container_width=True, hide_index=True)
        else:
            st.info(q2.get("error", ""))

    with st.expander("Q3: Which metrics move together? (Go to Statistical Visualisation for full heatmap)"):
        feat_cols = get_feature_columns(df)
        if len(feat_cols) >= 2:
            pearson = pearson_correlation(df, feat_cols)
            if not pearson.empty:
                st.markdown("**Pearson Correlation (top pairs):**")
                st.dataframe(pearson.round(3), use_container_width=True)
        else:
            st.info("At least 2 numeric feature columns needed for correlation.")

    # ── G2 ───────────────────────────────────────────────────────────────
    st.markdown("### G2 — Monitor Changes Across Builds")
    g2 = gqm_g2_monitoring()
    st.info(g2["description"])
    st.markdown("**Metrics:** " + ", ".join(g2["metrics"]))

    # ── G3 ───────────────────────────────────────────────────────────────
    st.markdown("### G3 — Evaluate the Decision Tree")
    g3 = gqm_g3_model(st.session_state.train_result)
    if "error" in g3:
        st.info(g3["error"])
    else:
        q1_g3 = g3.get("Q1", {})
        col1, col2, col3, col4 = st.columns(4)
        metric_card("Accuracy", f"{q1_g3.get('accuracy', 0):.3f}", col1)
        metric_card("Precision", f"{q1_g3.get('precision', 0):.3f}", col2)
        metric_card("Recall", f"{q1_g3.get('recall', 0):.3f}", col3)
        metric_card("F1-Score", f"{q1_g3.get('f1', 0):.3f}", col4)

    # ── G4 ───────────────────────────────────────────────────────────────
    st.markdown("### G4 — Evaluate Usability")
    g4 = gqm_g4_usability()
    st.markdown("**Usability tasks defined:**")
    for t in g4["tasks"]:
        st.markdown(f"• {t}")
    st.info("Record results on the **About / Usability** page.")

    # ── G5 ───────────────────────────────────────────────────────────────
    st.markdown("### G5 — Estimate Functional Size")
    if st.session_state.fp_result:
        fp = st.session_state.fp_result
        col1, col2, col3, col4 = st.columns(4)
        metric_card("UFP", fp.get("UFP", "N/A"), col1)
        metric_card("VAF", fp.get("VAF", "N/A"), col2)
        metric_card("AFP", fp.get("AFP", "N/A"), col3)
        metric_card("TDI", fp.get("TDI", "N/A"), col4)
    else:
        st.info("Go to **Function Point Analysis** to calculate functional size.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — STATISTICAL VISUALISATION
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Statistical Visualisation":
    st.title("📊 Statistical Visualisation")
    demo_banner()

    if st.session_state.df is None:
        st.warning("No dataset loaded.")
        st.stop()

    df = st.session_state.df
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if not c.startswith("_")]
    feature_cols = get_feature_columns(df)

    # ── A: Descriptive Stats ─────────────────────────────────────────────
    st.markdown("### A. Descriptive Statistics")
    stats_df = compute_descriptive_stats(df, feature_cols + ["Maintainability"])
    if not stats_df.empty:
        st.dataframe(stats_df.set_index("Metric").round(3), use_container_width=True)
    else:
        st.info("No numeric metric columns available.")

    # ── B: Box Plot ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### B. Box Plot")
    st.info("**IQR Rule:** Outliers are points beyond Q1 − 1.5×IQR or Q3 + 1.5×IQR (shown as orange dots).")

    if feature_cols:
        col1, col2 = st.columns(2)
        with col1:
            selected_metric = st.selectbox("Select metric", feature_cols, key="box_metric")
        with col2:
            group_by_risk = st.checkbox("Group by Risk label", value=("Risk" in df.columns), key="box_group")

        group_col = "Risk" if group_by_risk and "Risk" in df.columns else None
        box_png = plot_boxplot(df, selected_metric, group_col=group_col)
        if box_png:
            st.image(box_png, use_container_width=True)
            st.session_state.box_png = box_png
    else:
        st.info("No feature columns available for box plot.")

    # ── C: Control Chart ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### C. I-MR Control Chart")
    st.warning(
        "**Important:** A valid control chart requires **ordered build/release observations**. "
        "Using arbitrary row order as a substitute for build order produces meaningless results."
    )

    has_build_col = any(c.lower() in ["build", "release", "version", "sprint"] for c in df.columns)
    if has_build_col:
        build_candidates = [c for c in df.columns if c.lower() in ["build", "release", "version", "sprint"]]
        build_col = st.selectbox("Select build/order column", build_candidates)
        chart_metric = st.selectbox("Metric for control chart", feature_cols if feature_cols else ["CC"], key="cc_metric")
        df_sorted = df.sort_values(build_col)
        cc_series = df_sorted[chart_metric].dropna().reset_index(drop=True)
        cc_png = plot_imr_chart(cc_series, metric_name=chart_metric)
        if cc_png:
            st.image(cc_png, use_container_width=True)
    else:
        st.info(
            "No build/version column detected in your dataset. "
            "The control chart below uses the demo dataset's **row order** as an illustrative example only. "
            "This is NOT a valid statistical process control analysis."
        )
        if "CC" in df.columns:
            cc_series = df["CC"].dropna().reset_index(drop=True)
            cc_png = plot_imr_chart(cc_series, metric_name="CC (Demo – row order only)")
            if cc_png:
                st.image(cc_png, use_container_width=True)
                result = compute_imr_chart(cc_series)
                if "error" not in result:
                    ooc = result["out_of_control"]
                    if ooc:
                        st.warning(f"Example: {len(ooc)} point(s) outside control limits at positions: {ooc}")

    # ── D: Correlation ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### D. Correlation Analysis")
    st.caption("**Note:** Correlation measures linear association. Correlation does NOT imply causation.")

    corr_tab1, corr_tab2 = st.tabs(["Pearson Correlation", "Spearman Rank Correlation"])

    with corr_tab1:
        pearson_m = pearson_correlation(df, feature_cols)
        if not pearson_m.empty:
            corr_png = plot_correlation_heatmap(pearson_m, "Pearson Correlation Matrix")
            st.image(corr_png, use_container_width=True)
            st.session_state.corr_png = corr_png
            st.dataframe(pearson_m.round(3), use_container_width=True)
        else:
            st.info("Insufficient numeric columns.")

    with corr_tab2:
        spearman_m = spearman_correlation(df, feature_cols)
        if not spearman_m.empty:
            sp_png = plot_correlation_heatmap(spearman_m, "Spearman Rank Correlation Matrix")
            st.image(sp_png, use_container_width=True)
        else:
            st.info("Insufficient numeric columns.")

    # ── E: Group Comparison ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### E. Statistical Group Comparison (Mann–Whitney U Test)")

    if "Risk" in df.columns and feature_cols:
        metric_sel = st.selectbox("Select metric to test", feature_cols, key="mw_metric")
        mw = mann_whitney_test(df, metric_sel)
        if "error" in mw:
            st.warning(mw["error"])
        else:
            col1, col2, col3 = st.columns(3)
            metric_card("n (High)", mw["n_high"], col1)
            metric_card("n (Low)", mw["n_low"], col2)
            metric_card("p-value", f"{mw['p_value']:.4f}", col3)
            st.info(mw["interpretation"])
    else:
        st.info("Mann–Whitney test requires a 'Risk' label column and at least one feature column.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 6 — DECISION TREE & MODEL
# ════════════════════════════════════════════════════════════════════════════
elif page == "🌳 Decision Tree & Model":
    st.title("🌳 Decision Tree Classifier & Model Evaluation")
    demo_banner()

    if st.session_state.df is None:
        st.warning("No dataset loaded.")
        st.stop()

    df = st.session_state.df
    feature_cols = get_feature_columns(df)
    has_labels = "Risk" in df.columns and df["Risk"].notna().sum() > 0
    n_classes = len(df["Risk"].dropna().unique()) if has_labels else 0

    # ── Configuration ────────────────────────────────────────────────────
    st.markdown("### ⚙️ Model Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_features = st.multiselect(
            "Feature columns",
            feature_cols + (["Maintainability"] if "Maintainability" in df.columns else []),
            default=feature_cols,
        )
    with col2:
        max_depth = st.slider("Max tree depth", 1, 10, 3)
        min_leaf = st.slider("Min samples per leaf", 1, 10, 2)
    with col3:
        test_size = st.slider("Test set fraction", 0.1, 0.5, 0.25, 0.05)
        random_seed = st.number_input("Random seed", value=42, min_value=0)

    # ── Supervised mode ─────────────────────────────────────────────────
    if not has_labels or n_classes < 2:
        st.error(
            "❌ Supervised training is not available: "
            + ("No 'Risk' column found." if not has_labels else
               "Only one risk class present. Both 'High' and 'Low' are required.")
        )
        st.info(
            "💡 You can still explore **Heuristic Risk Scoring** on the Risk Prediction page "
            "without a labelled target."
        )
    else:
        if st.button("🚀 Train Decision Tree", type="primary"):
            if not selected_features:
                st.error("Select at least one feature column.")
            else:
                with st.spinner("Training model..."):
                    prep = prepare_data(df, selected_features)
                    if isinstance(prep, dict) and "error" in prep:
                        st.error(prep["error"])
                    else:
                        X, y, classes = prep
                        result = train_decision_tree(
                            X, y,
                            max_depth=int(max_depth),
                            min_samples_leaf=int(min_leaf),
                            test_size=float(test_size),
                            random_state=int(random_seed),
                        )
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            st.session_state.train_result = result
                            # Pre-generate charts
                            st.session_state.tree_png = plot_decision_tree(
                                result["clf"], result["feature_names"]
                            )
                            st.success("✅ Model trained successfully!")

        if st.session_state.train_result:
            r = st.session_state.train_result

            # ── Metrics ──────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 📈 Model Evaluation Results")
            st.caption(
                "These are measured values on the test split. "
                "Do not interpret demo-data accuracy as real-world performance."
            )
            col1, col2, col3, col4, col5 = st.columns(5)
            metric_card("Accuracy", f"{r['accuracy']:.3f}", col1)
            metric_card("Precision", f"{r['precision']:.3f}", col2)
            metric_card("Recall", f"{r['recall']:.3f}", col3)
            metric_card("F1-Score", f"{r['f1']:.3f}", col4)
            metric_card("Baseline", f"{r['baseline_accuracy']:.3f}", col5)

            if r["accuracy"] > r["baseline_accuracy"]:
                st.success(f"Model outperforms majority-class baseline ({r['baseline_accuracy']:.3f}) by {r['accuracy'] - r['baseline_accuracy']:.3f}.")
            else:
                st.warning("Model does not outperform the majority-class baseline. More data or better features may help.")

            if r.get("cv_results"):
                cv = r["cv_results"]
                st.info(
                    f"5-Fold Cross-Validation: Accuracy = {cv['cv_accuracy_mean']:.3f} ± {cv['cv_accuracy_std']:.3f}  |  "
                    f"F1 = {cv['cv_f1_mean']:.3f} ± {cv['cv_f1_std']:.3f}"
                )
            else:
                st.caption("Cross-validation skipped (dataset too small for 5-fold CV).")

            # ── Confusion Matrix ──────────────────────────────────────────
            st.markdown("#### Confusion Matrix")
            cm_png = plot_confusion_matrix(r["confusion_matrix"])
            col_l, col_r = st.columns([1, 2])
            col_l.image(cm_png)
            cm = np.array(r["confusion_matrix"])
            if cm.shape == (2, 2):
                col_r.markdown(f"""
| | Predicted Low | Predicted High |
|---|---|---|
| **Actual Low** | {cm[0,0]} ✅ TN | {cm[0,1]} ❌ FP |
| **Actual High** | {cm[1,0]} ❌ FN | {cm[1,1]} ✅ TP |
                """)

            # ── Classification report ─────────────────────────────────────
            st.markdown("#### Classification Report")
            cr = r["classification_report"]
            cr_df = pd.DataFrame(cr).T.round(3)
            st.dataframe(cr_df, use_container_width=True)

            # ── Feature importance ────────────────────────────────────────
            st.markdown("#### Feature Importance")
            fi_png = plot_feature_importance(r["feature_importances"])
            if fi_png:
                st.image(fi_png, use_container_width=True)

            # ── Decision tree diagram ─────────────────────────────────────
            st.markdown("#### Decision Tree Diagram")
            if st.session_state.tree_png:
                st.image(st.session_state.tree_png, use_container_width=True)
                st.download_button(
                    "⬇️ Download Tree Diagram",
                    data=st.session_state.tree_png,
                    file_name="decision_tree.png",
                    mime="image/png",
                )

            # ── Human-readable rules ──────────────────────────────────────
            st.markdown("#### Human-Readable Decision Rules")
            st.code(r["rules_text"], language="text")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 7 — RISK PREDICTION
# ════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Risk Prediction":
    st.title("🎯 Module-Level Risk Prediction")
    demo_banner()

    if st.session_state.df is None:
        st.warning("No dataset loaded.")
        st.stop()

    df = st.session_state.df
    feature_cols = get_feature_columns(df)
    has_model = st.session_state.train_result is not None

    if not has_model:
        st.info(
            "No trained model available. "
            "Train the Decision Tree on the **Decision Tree & Model** page, "
            "or use the **Heuristic Risk Scoring** below."
        )

    # ── Module-level prediction ──────────────────────────────────────────
    if has_model:
        st.markdown("### 🔍 Per-Module Prediction (Decision Tree)")
        r = st.session_state.train_result

        module_col = "Module" if "Module" in df.columns else None
        display_df = df.drop(columns=["_is_demo"], errors="ignore")

        all_preds = []
        pipe = r["pipeline"]
        feats = r["feature_names"]

        for _, row in display_df.iterrows():
            try:
                pred = predict_module(pipe, feats, row)
                all_preds.append({
                    "Module": row.get("Module", "—") if module_col else "—",
                    "Predicted Risk": pred["predicted_label"],
                    "Confidence %": pred["confidence_pct"],
                    "Actual Risk": row.get("Risk", "N/A"),
                })
            except Exception:
                all_preds.append({
                    "Module": row.get("Module", "—") if module_col else "—",
                    "Predicted Risk": "Error",
                    "Confidence %": None,
                    "Actual Risk": row.get("Risk", "N/A"),
                })

        pred_df = pd.DataFrame(all_preds)
        st.session_state.predictions_df = pred_df

        st.dataframe(pred_df, use_container_width=True, hide_index=True)

        # Single module drill-down
        st.markdown("#### 🔎 Module Drill-Down")
        if module_col:
            selected_mod = st.selectbox("Select a module", df["Module"].tolist())
            row = df[df["Module"] == selected_mod].iloc[0]
        else:
            idx = st.number_input("Select row index", 0, len(df) - 1, 0)
            row = df.iloc[idx]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Module Metrics:**")
            for feat in feats:
                val = row.get(feat, "N/A")
                st.markdown(f"- **{feat}:** {val}")
        with col2:
            pred = predict_module(pipe, feats, row)
            risk_color = "🔴" if pred["predicted_label"] == "High" else "🟢"
            st.markdown(f"**Predicted Risk:** {risk_color} **{pred['predicted_label']}**")
            if pred["confidence_pct"]:
                st.progress(int(pred["confidence_pct"]) / 100)
                st.caption(f"Confidence: {pred['confidence_pct']:.1f}%")
            st.markdown("**Decision Rules Applied:**")
            st.code(r["rules_text"], language="text")
            st.caption(
                "⚠️ Predicted risk is an indicator for human review, "
                "not verified proof of technical debt."
            )

        st.download_button(
            "⬇️ Download Predictions CSV",
            data=pred_df.to_csv(index=False),
            file_name="module_predictions.csv",
            mime="text/csv",
        )

    # ── Heuristic mode ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔢 Heuristic Risk Score")
    st.warning(
        "**Heuristic estimate — NOT an ML prediction.** "
        "Formula: weighted sum of min-max normalised LOC (30%), CC (30%), "
        "Dup (15%), Churn (15%), Coupling (10%). Score: 0–100."
    )

    if feature_cols:
        heuristic_scores = compute_heuristic_risk(df)
        heuristic_df = df[["Module"] + feature_cols].copy() if "Module" in df.columns else df[feature_cols].copy()
        heuristic_df["Heuristic Score (0-100)"] = heuristic_scores
        heuristic_df["Heuristic Label"] = heuristic_df["Heuristic Score (0-100)"].apply(
            lambda s: "High" if s >= 50 else "Low"
        )
        heuristic_df = heuristic_df.sort_values("Heuristic Score (0-100)", ascending=False)
        st.dataframe(heuristic_df.drop(columns=["_is_demo"], errors="ignore"), use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download Heuristic Scores CSV",
            data=heuristic_df.drop(columns=["_is_demo"], errors="ignore").to_csv(index=False),
            file_name="heuristic_risk_scores.csv",
            mime="text/csv",
        )
    else:
        st.info("No feature columns available for heuristic scoring.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 8 — REPORT & EXPORT
# ════════════════════════════════════════════════════════════════════════════
elif page == "📄 Report & Export":
    st.title("📄 Report & Export")
    demo_banner()

    if st.session_state.df is None:
        st.warning("No dataset loaded.")
        st.stop()

    df = st.session_state.df
    val = st.session_state.validation or {}
    summary = val.get("summary", {})
    feature_cols = get_feature_columns(df)
    stats_df = compute_descriptive_stats(df, feature_cols)

    st.markdown("### 📊 Available for Export")
    col1, col2, col3, col4 = st.columns(4)
    metric_card("Dataset Rows", summary.get("row_count", len(df)), col1)
    metric_card("Model Trained", "Yes" if st.session_state.train_result else "No", col2)
    metric_card("Predictions Ready", "Yes" if st.session_state.predictions_df is not None else "No", col3)
    metric_card("FP Analysis Done", "Yes" if st.session_state.fp_result else "No", col4)

    st.markdown("---")

    # ── Generate charts for report if not already cached ─────────────────
    if not st.session_state.corr_png and len(feature_cols) >= 2:
        pearson_m = pearson_correlation(df, feature_cols)
        if not pearson_m.empty:
            st.session_state.corr_png = plot_correlation_heatmap(pearson_m, "Pearson Correlation Matrix")

    if not st.session_state.box_png and feature_cols:
        st.session_state.box_png = plot_boxplot(df, feature_cols[0],
                                                  group_col="Risk" if "Risk" in df.columns else None)

    # ── PDF Report ───────────────────────────────────────────────────────
    st.markdown("### 📄 PDF Report")
    fp_combined = None
    if st.session_state.fp_result:
        fp_combined = st.session_state.fp_result

    if st.button("Generate PDF Report", type="primary"):
        with st.spinner("Building report..."):
            try:
                pdf_bytes = build_pdf_report(
                    dataset_name=st.session_state.dataset_name,
                    is_demo=st.session_state.is_demo,
                    validation_summary=summary,
                    stats_df=stats_df,
                    train_result=st.session_state.train_result,
                    fp_result=fp_combined,
                    predictions_df=st.session_state.predictions_df,
                    tree_png_bytes=st.session_state.tree_png,
                    corr_png_bytes=st.session_state.corr_png,
                    boxplot_png_bytes=st.session_state.box_png,
                )
                st.success("✅ Report generated!")
                st.download_button(
                    "⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"technical_debt_report_{datetime.date.today()}.pdf",
                    mime="application/pdf",
                )
            except Exception as ex:
                st.error(f"Error generating PDF: {ex}")

    st.markdown("---")

    # ── CSV Downloads ────────────────────────────────────────────────────
    st.markdown("### ⬇️ CSV Downloads")
    clean_csv = df.drop(columns=["_is_demo"], errors="ignore").to_csv(index=False)
    st.download_button("⬇️ Cleaned Dataset CSV", clean_csv, "cleaned_dataset.csv", "text/csv")

    if not stats_df.empty:
        st.download_button(
            "⬇️ Descriptive Statistics CSV",
            stats_df.to_csv(index=False),
            "descriptive_stats.csv",
            "text/csv",
        )

    if st.session_state.predictions_df is not None:
        st.download_button(
            "⬇️ Module Predictions CSV",
            st.session_state.predictions_df.to_csv(index=False),
            "module_predictions.csv",
            "text/csv",
        )

    if st.session_state.train_result:
        r = st.session_state.train_result
        eval_rows = [
            ("Accuracy", r["accuracy"]),
            ("Precision", r["precision"]),
            ("Recall", r["recall"]),
            ("F1-Score", r["f1"]),
            ("Baseline Accuracy", r["baseline_accuracy"]),
            ("Train Size", r["n_train"]),
            ("Test Size", r["n_test"]),
        ]
        eval_df = pd.DataFrame(eval_rows, columns=["Metric", "Value"])
        st.download_button(
            "⬇️ Model Evaluation Summary CSV",
            eval_df.to_csv(index=False),
            "model_evaluation.csv",
            "text/csv",
        )

    if st.session_state.tree_png:
        st.download_button(
            "⬇️ Decision Tree PNG",
            st.session_state.tree_png,
            "decision_tree.png",
            "image/png",
        )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 9 — FUNCTION POINT ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == "🖩 Function Point Analysis":
    st.title("🖩 Function Point Analysis (IFPUG)")
    demo_banner()
    st.markdown("""
    **IFPUG Function Point counting** estimates functional size from external inputs,
    outputs, inquiries, internal logical files, and external interface files.
    All values below are editable. Start with the example inventory or enter your own counts.
    """)

    # ── Function counts ──────────────────────────────────────────────────
    st.markdown("### Step 1: Enter Function Counts")
    demo_inv = get_demo_function_inventory()
    counts = {}

    for ftype, desc in [
        ("EI", "External Input (EI) — data entering the system"),
        ("EO", "External Output (EO) — data leaving the system"),
        ("EQ", "External Inquiry (EQ) — data retrieval without update"),
        ("ILF", "Internal Logical File (ILF) — data maintained by this system"),
        ("EIF", "External Interface File (EIF) — data from another system"),
    ]:
        st.markdown(f"**{desc}**")
        cols = st.columns(3)
        low = cols[0].number_input(f"{ftype} Low", 0, 999, demo_inv[ftype]["Low"], key=f"{ftype}_low")
        avg = cols[1].number_input(f"{ftype} Average", 0, 999, demo_inv[ftype]["Average"], key=f"{ftype}_avg")
        high = cols[2].number_input(f"{ftype} High", 0, 999, demo_inv[ftype]["High"], key=f"{ftype}_high")
        counts[ftype] = {"Low": int(low), "Average": int(avg), "High": int(high)}

    # ── UFP ──────────────────────────────────────────────────────────────
    ufp_result = calculate_ufp(counts)
    ufp = ufp_result["UFP"]

    st.markdown("---")
    st.markdown("### UFP Breakdown")
    rows = [["Function Type", "Low (×wt)", "Average (×wt)", "High (×wt)", "Subtotal FP"]]
    for ftype in ["EI", "EO", "EQ", "ILF", "EIF"]:
        b = ufp_result[ftype]["breakdown"]
        rows.append([
            ftype,
            f"{b['Low']['count']} × {b['Low']['weight']} = {b['Low']['fp']}",
            f"{b['Average']['count']} × {b['Average']['weight']} = {b['Average']['fp']}",
            f"{b['High']['count']} × {b['High']['weight']} = {b['High']['fp']}",
            str(ufp_result[ftype]["subtotal"]),
        ])
    rows.append(["**TOTAL UFP**", "", "", "", f"**{ufp}**"])
    st.table(pd.DataFrame(rows[1:], columns=rows[0]))

    # ── GSC Ratings ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Step 2: Rate the 14 General System Characteristics (0 = none, 5 = strong)")
    demo_gsc = get_demo_gsc_ratings()
    gsc_ratings = []
    for i, (name, default) in enumerate(zip(GSC_NAMES, demo_gsc)):
        r = st.slider(f"{i+1}. {name}", 0, 5, default, key=f"gsc_{i}")
        gsc_ratings.append(r)

    # ── VAF & AFP ────────────────────────────────────────────────────────
    st.markdown("---")
    vaf_result = calculate_vaf(gsc_ratings)
    afp = calculate_afp(ufp, vaf_result["VAF"])

    col1, col2, col3, col4 = st.columns(4)
    metric_card("UFP", ufp, col1)
    metric_card("TDI", vaf_result["TDI"], col2)
    metric_card("VAF", f"{vaf_result['VAF']:.4f}", col3)
    metric_card("AFP", afp, col4)

    st.latex(r"VAF = 0.65 + 0.01 \times TDI")
    st.latex(rf"VAF = 0.65 + 0.01 \times {vaf_result['TDI']} = {vaf_result['VAF']:.4f}")
    st.latex(rf"AFP = UFP \times VAF = {ufp} \times {vaf_result['VAF']:.4f} = {afp}")

    # ── Effort Estimate ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Step 3: Optional Effort Estimate")
    productivity = st.number_input("Productivity (FP per person-month)", 1.0, 1000.0, 10.0, 1.0)
    effort = estimate_effort(afp, productivity)
    st.info(
        f"Estimated effort: **{effort['effort_person_months']} person-months** "
        f"(AFP={afp} ÷ {productivity} FP/PM)\n\n"
        f"⚠️ {effort['disclaimer']}"
    )

    # Save for report
    st.session_state.fp_result = {
        "UFP": ufp, "TDI": vaf_result["TDI"],
        "VAF": vaf_result["VAF"], "AFP": afp,
        "effort": effort,
    }


# ════════════════════════════════════════════════════════════════════════════
# PAGE 10 — ABOUT / USABILITY
# ════════════════════════════════════════════════════════════════════════════
elif page == "👥 About / Usability":
    st.title("👥 About & Usability Evaluation")

    tab_about, tab_usability = st.tabs(["ℹ️ About", "🧪 Usability Evaluation"])

    with tab_about:
        st.markdown("""
        ## AI-Based Technical Debt Detection
        **Version:** 1.0 — College Project Prototype  
        **Purpose:** Educational demonstration of software metrics analysis and ML-based risk prediction.

        ### Technology Stack
        | Component | Library |
        |---|---|
        | Web UI | Streamlit |
        | Data processing | pandas, NumPy |
        | Machine learning | scikit-learn |
        | Charts | Matplotlib, Seaborn |
        | Statistics | SciPy |
        | PDF reports | ReportLab |
        | Interactive charts | Plotly |

        ### Important Disclaimers
        - Demo data is **synthetic**. Results must not be presented as real research.
        - Heuristic scores are **estimates**, not supervised ML predictions.
        - Model accuracy is measured on the available data and may not generalise.
        - Technical debt risk is an **indicator**, not a verified defect.
        - No source code or private data is sent to external services.

        ### Architecture
        ```
        technical-debt-detection/
        ├── app.py              ← Streamlit UI (this file)
        ├── data/               ← Demo CSV
        ├── src/
        │   ├── data_loader.py
        │   ├── validation.py
        │   ├── metrics_analysis.py
        │   ├── statistics_analysis.py
        │   ├── control_chart.py
        │   ├── model_training.py
        │   ├── gqm.py
        │   ├── function_points.py
        │   ├── report_generator.py
        │   └── usability.py
        └── tests/
        ```
        """)

    with tab_usability:
        st.markdown("### Usability Evaluation")
        st.info(
            "Record your observations for each task below. "
            "No participants are fabricated — only enter real data."
        )

        task_completions = []
        task_times = []
        task_errors_list = []

        for i, task in enumerate(TASKS):
            st.markdown(f"**Task {i+1}:** {task}")
            col1, col2, col3 = st.columns(3)
            completed = col1.checkbox("Completed", key=f"t_done_{i}")
            time_s = col2.number_input("Time (seconds)", 0, 9999, 0, key=f"t_time_{i}")
            errors = col3.number_input("Errors", 0, 99, 0, key=f"t_err_{i}")
            task_completions.append(completed)
            task_times.append(time_s)
            task_errors_list.append(errors)

        if any(task_completions):
            rate = calculate_completion_rate(task_completions)
            st.success(f"Task Completion Rate: **{rate}%** ({sum(task_completions)}/{len(task_completions)} tasks)")

        # ── SUS ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### System Usability Scale (SUS) — Optional")
        st.caption("Rate each statement 1 (Strongly Disagree) to 5 (Strongly Agree).")
        sus_responses = []
        for i, q in enumerate(SUS_QUESTIONS):
            r_sus = st.slider(f"{i+1}. {q}", 1, 5, 3, key=f"sus_{i}")
            sus_responses.append(r_sus)

        if st.button("Calculate SUS Score"):
            try:
                sus_result = calculate_sus(sus_responses)
                col1, col2, col3 = st.columns(3)
                metric_card("SUS Score", sus_result["sus_score"], col1)
                metric_card("Grade", sus_result["grade"], col2)
                metric_card("Rating", sus_result["adjective"], col3)
                st.info(sus_result["note"])
            except ValueError as ex:
                st.error(str(ex))
