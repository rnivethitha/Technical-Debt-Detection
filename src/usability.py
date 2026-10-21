# src/usability.py
# Records and evaluates usability tasks and optional SUS score.

import pandas as pd


TASKS = [
    "Upload a valid metrics CSV",
    "Find data-quality warnings",
    "Read a box plot",
    "Identify a point outside control limits",
    "Find module predictions and explain a decision rule",
    "Export a report",
]

# SUS scale questions (10 standard questions)
SUS_QUESTIONS = [
    "I think that I would like to use this system frequently.",
    "I found the system unnecessarily complex.",
    "I thought the system was easy to use.",
    "I think that I would need the support of a technical person to use this system.",
    "I found the various functions in this system were well integrated.",
    "I thought there was too much inconsistency in this system.",
    "I would imagine that most people would learn to use this system very quickly.",
    "I found the system very cumbersome to use.",
    "I felt very confident using the system.",
    "I needed to learn a lot of things before I could get going with this system.",
]

# Odd questions (1,3,5,7,9) contribute positively; even (2,4,6,8,10) negatively
SUS_POSITIVE_IDX = [0, 2, 4, 6, 8]   # 0-indexed
SUS_NEGATIVE_IDX = [1, 3, 5, 7, 9]


def calculate_sus(responses: list) -> dict:
    """Calculate SUS score from 10 Likert responses (each 1–5).

    Returns score (0–100), grade (A–F), and interpretation.
    """
    if len(responses) != 10:
        raise ValueError("Exactly 10 SUS responses required (1–5 each).")
    for r in responses:
        if not (1 <= r <= 5):
            raise ValueError(f"Each SUS response must be between 1 and 5. Got: {r}")

    score = 0
    for i in SUS_POSITIVE_IDX:
        score += (responses[i] - 1)
    for i in SUS_NEGATIVE_IDX:
        score += (5 - responses[i])
    sus_score = score * 2.5

    if sus_score >= 85:
        grade, adj = "A+", "Excellent"
    elif sus_score >= 80:
        grade, adj = "A", "Excellent"
    elif sus_score >= 70:
        grade, adj = "B", "Good"
    elif sus_score >= 58:
        grade, adj = "C", "OK"
    elif sus_score >= 51:
        grade, adj = "D", "Poor"
    else:
        grade, adj = "F", "Unacceptable"

    return {
        "sus_score": round(sus_score, 1),
        "grade": grade,
        "adjective": adj,
        "note": (
            "SUS score is based on self-reported responses. "
            "A score ≥ 68 is generally considered above average. "
            "Results from a single evaluator are indicative only."
        ),
    }


def calculate_completion_rate(completions: list) -> float:
    """Return task-completion percentage from list of True/False values."""
    if not completions:
        return 0.0
    return round(100 * sum(completions) / len(completions), 1)
