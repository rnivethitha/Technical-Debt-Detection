# src/function_points.py
# IFPUG Function Point Analysis calculator.
# Based on the standard IFPUG counting rules (weights and VAF formula).

import pandas as pd

# IFPUG weights table
WEIGHTS = {
    "EI":  {"Low": 3, "Average": 4, "High": 6},   # External Input
    "EO":  {"Low": 4, "Average": 5, "High": 7},   # External Output
    "EQ":  {"Low": 3, "Average": 4, "High": 6},   # External Inquiry
    "ILF": {"Low": 7, "Average": 10, "High": 15}, # Internal Logical File
    "EIF": {"Low": 5, "Average": 7, "High": 10},  # External Interface File
}

# The 14 General System Characteristics (GSC) for VAF
GSC_NAMES = [
    "Data Communications",
    "Distributed Data Processing",
    "Performance",
    "Heavily Used Configuration",
    "Transaction Rate",
    "Online Data Entry",
    "End-User Efficiency",
    "Online Update",
    "Complex Processing",
    "Reusability",
    "Installation Ease",
    "Operational Ease",
    "Multiple Sites",
    "Facilitate Change",
]


def calculate_ufp(counts: dict) -> dict:
    """Calculate Unadjusted Function Points from counts.

    counts format:
        {
          "EI": {"Low": 2, "Average": 3, "High": 1},
          "EO": {"Low": 1, "Average": 2, "High": 0},
          ...
        }
    Returns dict with per-type UFP and total UFP.
    """
    result = {}
    total = 0
    for ftype, complexity_counts in counts.items():
        if ftype not in WEIGHTS:
            continue
        type_total = 0
        breakdown = {}
        for complexity, count in complexity_counts.items():
            weight = WEIGHTS[ftype].get(complexity, 0)
            fp = count * weight
            breakdown[complexity] = {"count": count, "weight": weight, "fp": fp}
            type_total += fp
        result[ftype] = {"breakdown": breakdown, "subtotal": type_total}
        total += type_total
    result["UFP"] = total
    return result


def calculate_vaf(gsc_ratings: list) -> dict:
    """Calculate Value Adjustment Factor from 14 GSC ratings (each 0–5).

    VAF = 0.65 + 0.01 × TDI
    AFP = UFP × VAF
    """
    if len(gsc_ratings) != 14:
        raise ValueError("Exactly 14 GSC ratings are required.")
    for r in gsc_ratings:
        if not (0 <= r <= 5):
            raise ValueError(f"Each GSC rating must be between 0 and 5. Got: {r}")
    tdi = sum(gsc_ratings)
    vaf = 0.65 + 0.01 * tdi
    return {"TDI": tdi, "VAF": round(vaf, 4)}


def calculate_afp(ufp: int, vaf: float) -> float:
    """AFP = UFP × VAF."""
    return round(ufp * vaf, 2)


def estimate_effort(afp: float, productivity: float = 10.0) -> dict:
    """Optional effort estimate.

    productivity: function points per person-month (user-editable assumption).
    Returns person-months and a disclaimer.
    """
    effort_pm = round(afp / productivity, 2)
    return {
        "afp": afp,
        "productivity_fp_per_pm": productivity,
        "effort_person_months": effort_pm,
        "disclaimer": (
            "This is an indicative estimate only. "
            "Productivity varies significantly by team, technology, and domain. "
            "Validate against your organisation's historical data."
        ),
    }


def get_demo_function_inventory() -> dict:
    """Return the example function inventory from the report's typical structure.
    
    Users can edit these values in the UI. This is NOT a fixed result.
    """
    return {
        "EI":  {"Low": 3, "Average": 4, "High": 2},
        "EO":  {"Low": 2, "Average": 3, "High": 1},
        "EQ":  {"Low": 4, "Average": 3, "High": 1},
        "ILF": {"Low": 2, "Average": 2, "High": 1},
        "EIF": {"Low": 1, "Average": 2, "High": 0},
    }


def get_demo_gsc_ratings() -> list:
    """Return illustrative GSC ratings (all 3 = moderate). Users can adjust."""
    return [3, 3, 3, 2, 3, 4, 3, 3, 2, 3, 2, 3, 2, 3]
