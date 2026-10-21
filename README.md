# AI-Based Technical Debt Detection

> **Software Metrics Analysis, Risk Prediction and Decision Support**  
> College Project Prototype — Python / Streamlit

---

## Quick Start (Windows)

### 1. Open the project folder

Open **Command Prompt** or **PowerShell** and navigate to the project:

```
cd C:\Users\rnive\Downloads\sm
```

### 2. Install dependencies

```
pip install --user -r requirements.txt
```

### 3. Run the application

```
python -m streamlit run app.py
```

### 4. Open in your browser

Streamlit will print a URL like:

```
Local URL: http://localhost:8501
```

Open that URL in **Chrome** or **Edge**.

---

## Demo Walkthrough (for Faculty Demonstration)

1. Click **Dataset Upload** in the sidebar.
2. Click **Load Demo Dataset**.
3. Go to **Data Quality** — review warnings and summary.
4. Go to **Statistical Visualisation** — view box plots and correlation heatmap.
5. Go to **Decision Tree & Model** — click **Train Decision Tree**.
6. Go to **Risk Prediction** — view per-module predictions and drill-down.
7. Go to **Function Point Analysis** — adjust counts and view AFP.
8. Go to **Report & Export** — click **Generate PDF Report** and download.

> ⚠️ **DEMO DATA:** Results from the demo dataset are illustrative only.  
> Do NOT present them as real-world experimental findings.

---

## Uploading Your Own CSV

1. Go to **Dataset Upload → Mode B**.
2. Upload your CSV file.
3. Use the **column mapping** dropdowns to map your column names to:
   - Module, LOC, CC, Dup, Churn, Coupling, Risk (optional)
4. Click **Apply Mapping & Validate**.

---

## Running Tests

```
python -m pytest tests/ -v
```

---

## Project Structure

```
technical-debt-detection/
├── app.py                    ← Main Streamlit application
├── requirements.txt
├── README.md
├── data/
│   └── demo_modules.csv      ← 30 synthetic demo modules
├── src/
│   ├── __init__.py
│   ├── data_loader.py        ← CSV loading and column mapping
│   ├── validation.py         ← Dataset validation and cleaning
│   ├── metrics_analysis.py   ← Metric definitions, descriptive stats, heuristic risk
│   ├── statistics_analysis.py← Pearson, Spearman, Mann-Whitney, box plots
│   ├── control_chart.py      ← I-MR control chart
│   ├── model_training.py     ← Decision tree, pipeline, evaluation
│   ├── gqm.py                ← GQM framework
│   ├── function_points.py    ← IFPUG Function Point Analysis
│   ├── report_generator.py   ← PDF report (ReportLab)
│   └── usability.py          ← SUS and task completion
├── tests/
│   ├── test_validation.py
│   ├── test_statistics.py
│   ├── test_model.py
│   └── test_function_points.py
└── outputs/                  ← Download destination for exports
```

---

## Application Pages

| Page | What it does |
|------|-------------|
| 🏠 Overview | Dashboard with key metrics |
| 📤 Dataset Upload | Load demo or upload CSV with column mapping |
| ✅ Data Quality | Errors, warnings, missing values, class distribution |
| 📐 GQM & Metric Analysis | Goal–Question–Metric framework results |
| 📊 Statistical Visualisation | Box plots, control charts, correlation heatmaps, Mann-Whitney |
| 🌳 Decision Tree & Model | Train, evaluate, visualise the classifier |
| 🎯 Risk Prediction | Per-module predictions + heuristic scoring |
| 📄 Report & Export | PDF + CSV downloads |
| 🖩 Function Point Analysis | IFPUG UFP, VAF, AFP, effort estimate |
| 👥 About / Usability | SUS evaluation, task completion recording |

---

## Common Errors and Fixes

| Error | Fix |
|-------|-----|
| `streamlit: command not found` | Use `python -m streamlit run app.py` |
| `ModuleNotFoundError: streamlit` | Run `pip install --user -r requirements.txt` |
| `Only one Risk class` | Your dataset needs both High and Low labels for supervised ML |
| PDF generation fails | Ensure `reportlab` is installed |
| Port already in use | Add `--server.port 8502` to the run command |

---

## Important Notes

- **Demo data is synthetic** — created for demonstration purposes only.
- **Heuristic risk scores** use a documented weighted formula, NOT ML.
- **Model accuracy** is measured on your data only; do not over-generalise.
- No source code or private data is sent to external services.
- All processing happens locally on your machine.
