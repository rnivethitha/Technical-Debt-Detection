# tests/test_function_points.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.function_points import (
    calculate_ufp, calculate_vaf, calculate_afp,
    estimate_effort, get_demo_function_inventory, get_demo_gsc_ratings
)


# ── UFP calculation ──────────────────────────────────────────────────────
def test_ufp_basic_calculation():
    counts = {
        "EI":  {"Low": 1, "Average": 0, "High": 0},  # 1×3 = 3
        "EO":  {"Low": 0, "Average": 0, "High": 0},
        "EQ":  {"Low": 0, "Average": 0, "High": 0},
        "ILF": {"Low": 0, "Average": 0, "High": 0},
        "EIF": {"Low": 0, "Average": 0, "High": 0},
    }
    result = calculate_ufp(counts)
    assert result["UFP"] == 3


def test_ufp_zero_counts():
    counts = {ft: {"Low": 0, "Average": 0, "High": 0} for ft in ["EI", "EO", "EQ", "ILF", "EIF"]}
    result = calculate_ufp(counts)
    assert result["UFP"] == 0


def test_ufp_demo_inventory_is_positive():
    inv = get_demo_function_inventory()
    result = calculate_ufp(inv)
    assert result["UFP"] > 0


def test_ufp_ilf_weights():
    counts = {
        "EI":  {"Low": 0, "Average": 0, "High": 0},
        "EO":  {"Low": 0, "Average": 0, "High": 0},
        "EQ":  {"Low": 0, "Average": 0, "High": 0},
        "ILF": {"Low": 1, "Average": 1, "High": 1},  # 7+10+15 = 32
        "EIF": {"Low": 0, "Average": 0, "High": 0},
    }
    result = calculate_ufp(counts)
    assert result["ILF"]["subtotal"] == 32


# ── VAF calculation ──────────────────────────────────────────────────────
def test_vaf_all_zeros():
    ratings = [0] * 14
    result = calculate_vaf(ratings)
    assert result["TDI"] == 0
    assert abs(result["VAF"] - 0.65) < 1e-9


def test_vaf_all_fives():
    ratings = [5] * 14
    result = calculate_vaf(ratings)
    assert result["TDI"] == 70
    assert abs(result["VAF"] - 1.35) < 1e-9


def test_vaf_wrong_count_raises():
    with pytest.raises(ValueError):
        calculate_vaf([3] * 13)


def test_vaf_out_of_range_raises():
    with pytest.raises(ValueError):
        calculate_vaf([3] * 13 + [6])  # 6 is invalid


def test_vaf_demo_ratings():
    ratings = get_demo_gsc_ratings()
    result = calculate_vaf(ratings)
    assert 0.65 <= result["VAF"] <= 1.35


# ── AFP ───────────────────────────────────────────────────────────────────
def test_afp_formula():
    afp = calculate_afp(100, 1.00)
    assert afp == 100.0


def test_afp_zero_ufp():
    afp = calculate_afp(0, 1.20)
    assert afp == 0.0


# ── Effort ────────────────────────────────────────────────────────────────
def test_effort_estimate():
    result = estimate_effort(100.0, 10.0)
    assert result["effort_person_months"] == 10.0
    assert "disclaimer" in result


def test_effort_high_productivity():
    result = estimate_effort(50.0, 25.0)
    assert result["effort_person_months"] == 2.0
