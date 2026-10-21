# src/control_chart.py
# Individuals and Moving Range (I-MR) control chart implementation.
# Requires ordered build/release data. Do not use arbitrary row index as build order.

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io


def compute_imr_chart(series: pd.Series) -> dict:
    """Compute I-MR control chart statistics for an ordered series of observations.

    Returns a dict with:
        values, moving_range, cl (center line), ucl, lcl,
        mr_cl, mr_ucl, out_of_control indices and their values.

    Constants for I-MR (n=2 subgroup):
        d2 = 1.128,  D3 = 0 (lower MR limit),  D4 = 3.267
    """
    values = series.dropna().values
    n = len(values)
    if n < 4:
        return {"error": "At least 4 ordered observations are required for an I-MR chart."}

    # Moving ranges (absolute differences between consecutive observations)
    mr = np.abs(np.diff(values))

    # Constants for subgroup of 2
    d2 = 1.128
    D4 = 3.267
    # D3 = 0 (lower control limit for MR is always 0 for n=2)

    mr_bar = np.mean(mr)
    x_bar = np.mean(values)

    # Individuals chart limits
    ucl = x_bar + 3 * (mr_bar / d2)
    lcl = x_bar - 3 * (mr_bar / d2)

    # MR chart limits
    mr_ucl = D4 * mr_bar

    # Points outside control limits (individuals chart)
    ooc_indices = [i for i, v in enumerate(values) if v > ucl or v < lcl]

    return {
        "values": values.tolist(),
        "moving_range": mr.tolist(),
        "cl": float(x_bar),
        "ucl": float(ucl),
        "lcl": float(lcl),
        "mr_cl": float(mr_bar),
        "mr_ucl": float(mr_ucl),
        "out_of_control": ooc_indices,
        "n": n,
    }


def plot_imr_chart(
    series: pd.Series,
    metric_name: str = "CC",
    build_labels: list | None = None,
) -> bytes:
    """Generate I-MR control chart and return PNG bytes.

    series    – ordered metric values (one per build)
    build_labels – optional list of build identifiers for x-axis
    """
    result = compute_imr_chart(series)
    if "error" in result:
        return b""

    values = np.array(result["values"])
    mr = np.array(result["moving_range"])
    n = result["n"]
    x_labels = build_labels if build_labels else [str(i + 1) for i in range(n)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    fig.suptitle(
        f"I-MR Control Chart — {metric_name}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # ── Individuals chart ──────────────────────────────────────────────────
    ax1.plot(range(n), values, marker="o", color="#2980b9", linewidth=1.5, zorder=3)
    ax1.axhline(result["cl"], color="green", linewidth=1.5, linestyle="-", label=f"CL={result['cl']:.2f}")
    ax1.axhline(result["ucl"], color="red", linewidth=1.2, linestyle="--", label=f"UCL={result['ucl']:.2f}")
    ax1.axhline(result["lcl"], color="red", linewidth=1.2, linestyle="--", label=f"LCL={result['lcl']:.2f}")

    for idx in result["out_of_control"]:
        ax1.plot(idx, values[idx], "ro", markersize=10, zorder=5, label="_nolegend_")
        ax1.annotate("OOC", (idx, values[idx]), textcoords="offset points", xytext=(5, 5), color="red", fontsize=8)

    ax1.set_ylabel(metric_name, fontsize=10)
    ax1.set_title("Individuals (I) Chart", fontsize=10)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)

    # ── Moving Range chart ─────────────────────────────────────────────────
    ax2.plot(range(1, n), mr, marker="s", color="#8e44ad", linewidth=1.5, zorder=3)
    ax2.axhline(result["mr_cl"], color="green", linewidth=1.5, linestyle="-", label=f"MR̄={result['mr_cl']:.2f}")
    ax2.axhline(result["mr_ucl"], color="red", linewidth=1.2, linestyle="--", label=f"UCL={result['mr_ucl']:.2f}")
    ax2.axhline(0, color="red", linewidth=1.2, linestyle="--", label="LCL=0")
    ax2.set_ylabel("Moving Range", fontsize=10)
    ax2.set_title("Moving Range (MR) Chart", fontsize=10)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.set_xticks(range(1, n))
    ax2.set_xticklabels(x_labels[1:], rotation=45, ha="right", fontsize=7)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
