# src/report_generator.py
# Generates a PDF report using ReportLab.

import io
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, PageBreak,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT


NAVY = colors.HexColor("#1a2744")
ACCENT = colors.HexColor("#2980b9")
LIGHT_GREY = colors.HexColor("#f4f6fa")
RED = colors.HexColor("#E05252")
GREEN = colors.HexColor("#27ae60")


def _style(name, parent="Normal", **kwargs):
    styles = getSampleStyleSheet()
    base = styles.get(parent, styles["Normal"])
    return ParagraphStyle(name=name, parent=base, **kwargs)


def build_pdf_report(
    dataset_name: str,
    is_demo: bool,
    validation_summary: dict,
    stats_df,
    train_result: dict | None,
    fp_result: dict | None,
    predictions_df=None,
    tree_png_bytes: bytes | None = None,
    corr_png_bytes: bytes | None = None,
    boxplot_png_bytes: bytes | None = None,
) -> bytes:
    """Build a complete PDF report and return as bytes."""

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    story = []
    styles = getSampleStyleSheet()

    title_style = _style("Title", "Title", fontSize=18, textColor=NAVY, alignment=TA_CENTER, spaceAfter=6)
    subtitle_style = _style("Subtitle", "Normal", fontSize=11, textColor=ACCENT, alignment=TA_CENTER, spaceAfter=16)
    h1_style = _style("H1", "Heading1", fontSize=14, textColor=NAVY, spaceBefore=14, spaceAfter=4)
    h2_style = _style("H2", "Heading2", fontSize=11, textColor=ACCENT, spaceBefore=10, spaceAfter=4)
    body_style = _style("Body", "Normal", fontSize=9, spaceAfter=4)
    warn_style = _style("Warn", "Normal", fontSize=9, textColor=RED, spaceAfter=4)
    small_style = _style("Small", "Normal", fontSize=8, textColor=colors.grey, spaceAfter=4)

    # ── Title block ─────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("AI-Based Technical Debt Detection", title_style))
    story.append(Paragraph("Software Metrics Analysis, Risk Prediction and Decision Support", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=NAVY))
    story.append(Spacer(1, 0.3 * cm))
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    story.append(Paragraph(f"Generated: {now}", small_style))
    story.append(Spacer(1, 0.5 * cm))

    # ── Demo notice ─────────────────────────────────────────────────────────
    if is_demo:
        story.append(Paragraph(
            "⚠ DEMO DATA: Results in this report are illustrative only and must NOT be "
            "presented as real-world experimental findings.",
            warn_style,
        ))
        story.append(Spacer(1, 0.3 * cm))

    # ── Dataset info ────────────────────────────────────────────────────────
    story.append(Paragraph("1. Dataset Information", h1_style))
    ds_rows = [
        ["Dataset name", dataset_name],
        ["Source", "Demo (Synthetic)" if is_demo else "User-uploaded CSV"],
        ["Rows", str(validation_summary.get("row_count", "N/A"))],
        ["Columns", str(validation_summary.get("col_count", "N/A"))],
        ["Total missing values", str(validation_summary.get("total_missing", "N/A"))],
        ["Duplicate rows", str(validation_summary.get("duplicate_rows", "N/A"))],
    ]
    risk_info = validation_summary.get("risk", {})
    if risk_info.get("class_distribution"):
        for cls, cnt in risk_info["class_distribution"].items():
            ds_rows.append([f"Risk = {cls}", str(cnt)])
    story.append(_make_table(ds_rows))
    story.append(Spacer(1, 0.4 * cm))

    # ── Data quality ─────────────────────────────────────────────────────────
    story.append(Paragraph("2. Data Quality", h1_style))
    if validation_summary.get("missing_by_column"):
        story.append(Paragraph("Missing values by column:", h2_style))
        mv_rows = [[col, str(cnt)] for col, cnt in validation_summary["missing_by_column"].items()]
        story.append(_make_table([["Column", "Missing Count"]] + mv_rows))
    else:
        story.append(Paragraph("No missing values detected.", body_style))
    story.append(Spacer(1, 0.4 * cm))

    # ── Descriptive statistics ───────────────────────────────────────────────
    story.append(Paragraph("3. Descriptive Statistics", h1_style))
    if stats_df is not None and not stats_df.empty:
        cols = list(stats_df.columns)
        rows = [cols] + stats_df.astype(str).values.tolist()
        story.append(_make_table(rows))
    else:
        story.append(Paragraph("No statistics available.", body_style))
    story.append(Spacer(1, 0.4 * cm))

    # ── Charts ───────────────────────────────────────────────────────────────
    story.append(Paragraph("4. Charts", h1_style))
    for label, png_bytes in [
        ("Box Plot", boxplot_png_bytes),
        ("Correlation Heatmap", corr_png_bytes),
    ]:
        if png_bytes:
            story.append(Paragraph(label, h2_style))
            img = RLImage(io.BytesIO(png_bytes), width=14 * cm, height=8 * cm)
            story.append(img)
            story.append(Spacer(1, 0.3 * cm))

    # ── Model evaluation ─────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("5. Decision-Tree Model Evaluation", h1_style))
    if train_result:
        model_rows = [
            ["Metric", "Value"],
            ["Accuracy", f"{train_result['accuracy']:.4f}"],
            ["Precision", f"{train_result['precision']:.4f}"],
            ["Recall", f"{train_result['recall']:.4f}"],
            ["F1-Score", f"{train_result['f1']:.4f}"],
            ["Majority-class baseline", f"{train_result['baseline_accuracy']:.4f}"],
            ["Training samples", str(train_result['n_train'])],
            ["Test samples", str(train_result['n_test'])],
        ]
        cv = train_result.get("cv_results", {})
        if cv:
            model_rows.append(["CV Accuracy (5-fold mean ± std)",
                                f"{cv['cv_accuracy_mean']:.4f} ± {cv['cv_accuracy_std']:.4f}"])
        story.append(_make_table(model_rows))
        story.append(Spacer(1, 0.4 * cm))

        story.append(Paragraph("Feature Importances:", h2_style))
        fi_rows = [["Feature", "Importance"]] + [
            [k, f"{v:.4f}"] for k, v in sorted(train_result["feature_importances"].items(), key=lambda x: -x[1])
        ]
        story.append(_make_table(fi_rows))
        story.append(Spacer(1, 0.4 * cm))

        story.append(Paragraph("Human-Readable Decision Rules:", h2_style))
        rules = train_result.get("rules_text", "Not available.")
        story.append(Paragraph(f"<font name='Courier' size='7'>{rules.replace(chr(10), '<br/>')}</font>", body_style))
        story.append(Spacer(1, 0.4 * cm))

        if tree_png_bytes:
            story.append(Paragraph("Decision Tree Diagram:", h2_style))
            img = RLImage(io.BytesIO(tree_png_bytes), width=16 * cm, height=7 * cm)
            story.append(img)
    else:
        story.append(Paragraph("No model trained. Train on the Decision Tree page.", body_style))

    # ── Function Points ───────────────────────────────────────────────────────
    if fp_result:
        story.append(PageBreak())
        story.append(Paragraph("6. Function Point Analysis", h1_style))
        fp_rows = [
            ["UFP", str(fp_result.get("UFP", "N/A"))],
            ["TDI", str(fp_result.get("TDI", "N/A"))],
            ["VAF", str(fp_result.get("VAF", "N/A"))],
            ["AFP", str(fp_result.get("AFP", "N/A"))],
        ]
        if "effort" in fp_result:
            fp_rows.append(["Effort (person-months)", str(fp_result["effort"].get("effort_person_months", "N/A"))])
        story.append(_make_table(fp_rows))

    # ── Module predictions ────────────────────────────────────────────────────
    if predictions_df is not None and not predictions_df.empty:
        story.append(PageBreak())
        story.append(Paragraph("7. Module-Level Predictions", h1_style))
        story.append(Paragraph(
            "IMPORTANT: Predicted risk class is an indicator for human review, "
            "not proof of a defect.",
            warn_style,
        ))
        pcols = list(predictions_df.columns[:6])  # limit width
        prows = [pcols] + predictions_df[pcols].astype(str).head(30).values.tolist()
        story.append(_make_table(prows))

    # ── Limitations ──────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph("8. Limitations and Interpretation", h1_style))
    story.append(Paragraph(
        "• Risk labels are illustrative indicators, not verified defect reports. "
        "• Model accuracy depends entirely on data quality and label reliability. "
        "• Demo-mode results are synthetic and must not be cited as research findings. "
        "• Heuristic scores use documented formulas but are not equivalent to supervised ML. "
        "• Always validate automated findings with domain expert review.",
        body_style,
    ))

    doc.build(story)
    return buf.getvalue()


def _make_table(rows: list) -> Table:
    """Create a styled ReportLab table from a list of rows."""
    t = Table(rows, repeatRows=1)
    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("BACKGROUND", (0, 1), (-1, -1), LIGHT_GREY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GREY, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ])
    t.setStyle(style)
    return t
