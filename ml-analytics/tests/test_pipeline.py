"""
test_pipeline.py — Unit tests for the ML Analytics pipeline.

Tests cover:
    1. Valid data — happy path
    2. Missing values — validation removes them
    3. Invalid numeric values — coerced and removed
    4. Duplicate records — deduplicated
    5. Range violations — out-of-range rows removed
    6. Baseline calculation — correct mean from previous sessions
    7. Insufficient history — baseline_available = False
    8. Behavioral score — in valid range [0, 1]
    9. Trend detection — correct labels for known patterns
    10. JSON output — valid structure, required keys present

Run with:
    python3 -m pytest tests/test_pipeline.py -v

Project: AI-Based Cognitive Gaming and Memory Assistance Platform
SIH 2026 | Problem ID: SIH26003 | Team ID: 271
"""

import sys
import os
import json
import math
import tempfile
import pytest
import pandas as pd
import numpy as np

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import load_telemetry, validate, preprocess, engineer_features
from src.baseline      import compute_personal_baseline
from src.scoring       import compute_scores
from src.analytics     import detect_trends, build_results
from src.config        import MIN_BASELINE_SESSIONS


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _make_row(
    session_id="S001", user_id="U001", timestamp="2026-01-01 10:00:00",
    touch_drift=20.0, response_latency=1000.0, decision_speed=3.0,
    mistakes=3, completion_rate=0.85, session_duration=500.0,
    game_type="memory_match", data_note="SYNTHETIC",
):
    return {
        "session_id": session_id, "user_id": user_id, "timestamp": timestamp,
        "touch_drift": touch_drift, "response_latency": response_latency,
        "decision_speed": decision_speed, "mistakes": mistakes,
        "completion_rate": completion_rate, "session_duration": session_duration,
        "game_type": game_type, "data_note": data_note,
    }


def _make_df(*rows):
    return pd.DataFrame(list(rows))


def _full_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the complete pipeline and return the final DataFrame."""
    df, _ = validate(df)
    df = preprocess(df)
    df = engineer_features(df)
    df = compute_personal_baseline(df)
    df = compute_scores(df)
    df = detect_trends(df)
    return df


def _multi_session_df(user_id="U001", n=6, **kwargs):
    """Build n chronological sessions for one user."""
    rows = []
    for i in range(n):
        rows.append(_make_row(
            session_id=f"{user_id}_S{i+1:02d}",
            user_id=user_id,
            timestamp=f"2026-0{max(1, (i//30)+1):02d}-{(i % 28)+1:02d} 10:00:00",
            **kwargs,
        ))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# TEST 1: Valid data — happy path
# ─────────────────────────────────────────────────────────────

def test_valid_data_loads():
    """All 44 rows of the synthetic dataset should pass validation."""
    df = load_telemetry("data/telemetry_sample.csv")
    df_valid, issues = validate(df)
    assert len(df_valid) == len(df), "All rows should pass for the clean synthetic dataset."
    assert issues == [], f"Expected no issues, got: {issues}"


def test_valid_data_full_pipeline():
    """Full pipeline should run without exceptions on clean data."""
    df = load_telemetry("data/telemetry_sample.csv")
    df = _full_pipeline(df)
    assert len(df) == 44
    assert "behavioral_score" in df.columns
    assert "trend" in df.columns


# ─────────────────────────────────────────────────────────────
# TEST 2: Missing values — validation removes rows
# ─────────────────────────────────────────────────────────────

def test_missing_value_removal():
    """Rows with missing required fields should be removed during validation."""
    rows = [_make_row("S001", "U001"), _make_row("S002", "U001")]
    df = pd.DataFrame(rows)
    df.loc[1, "response_latency"] = None   # Inject missing value

    df_valid, issues = validate(df)
    assert len(df_valid) == 1, "Row with missing value should be removed."
    assert any("missing" in i.lower() for i in issues)


# ─────────────────────────────────────────────────────────────
# TEST 3: Invalid numeric values — coerced to NaN and removed
# ─────────────────────────────────────────────────────────────

def test_invalid_numeric_removal():
    """Rows with non-numeric values in numeric columns should be removed."""
    good_row = _make_row("S001", "U001")
    # Build the bad row explicitly so the column is object-typed from the start,
    # avoiding the pandas FutureWarning about incompatible dtype assignment via loc.
    bad_row = _make_row("S002", "U001")
    bad_row["touch_drift"] = "not_a_number"   # dict mutation — no dtype conflict

    df = pd.DataFrame([good_row, bad_row])     # mixed dtype → object column

    df_valid, issues = validate(df)
    assert len(df_valid) == 1, "Row with non-numeric value should be removed."


# ─────────────────────────────────────────────────────────────
# TEST 4: Duplicate records — deduplicated
# ─────────────────────────────────────────────────────────────

def test_duplicate_removal():
    """Duplicate session_ids should be deduplicated (keep first)."""
    rows = [_make_row("S001", "U001"), _make_row("S001", "U001")]
    df = pd.DataFrame(rows)

    df_valid, issues = validate(df)
    assert len(df_valid) == 1, "Duplicate session should be removed."
    assert any("duplicate" in i.lower() for i in issues)


# ─────────────────────────────────────────────────────────────
# TEST 5: Range violations — out-of-range rows removed
# ─────────────────────────────────────────────────────────────

def test_range_violation_removal():
    """Rows with values outside valid ranges should be removed."""
    rows = [_make_row("S001", "U001"), _make_row("S002", "U001", completion_rate=1.5)]
    df = pd.DataFrame(rows)

    df_valid, issues = validate(df)
    assert len(df_valid) == 1, "Row with out-of-range completion_rate should be removed."


# ─────────────────────────────────────────────────────────────
# TEST 6: Baseline calculation — correct mean from previous sessions
# ─────────────────────────────────────────────────────────────

def test_baseline_calculation_correct():
    """
    Baseline for session N must equal the mean of sessions 1..(N-1).
    We test this with controlled, known values.
    """
    known_drift = [10.0, 20.0, 30.0, 40.0]  # sessions 1–4
    rows = []
    for i, d in enumerate(known_drift):
        rows.append(_make_row(
            session_id=f"S{i+1:02d}", user_id="U001",
            timestamp=f"2026-01-0{i+1} 10:00:00",
            touch_drift=d,
        ))
    df = pd.DataFrame(rows)
    df, _ = validate(df)
    df = preprocess(df)
    df = engineer_features(df)
    df = compute_personal_baseline(df)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Session 4 (index 3) should have baseline = mean(10, 20, 30) = 20.0
    session4 = df[df["session_id"] == "S04"].iloc[0]
    assert session4["baseline_available"] == True
    assert abs(session4["baseline_touch_drift"] - 20.0) < 0.01, (
        f"Expected baseline_touch_drift ≈ 20.0, got {session4['baseline_touch_drift']}"
    )


# ─────────────────────────────────────────────────────────────
# TEST 7: Insufficient history — baseline_available = False
# ─────────────────────────────────────────────────────────────

def test_insufficient_history():
    """Users with fewer than MIN_BASELINE_SESSIONS should have baseline_available = False."""
    df = _multi_session_df("U001", n=MIN_BASELINE_SESSIONS - 1)
    df, _ = validate(df)
    df = preprocess(df)
    df = engineer_features(df)
    df = compute_personal_baseline(df)

    assert df["baseline_available"].sum() == 0, (
        f"Expected 0 sessions with baseline, got {df['baseline_available'].sum()}"
    )


# ─────────────────────────────────────────────────────────────
# TEST 8: Behavioral score — in valid range [0, 1]
# ─────────────────────────────────────────────────────────────

def test_score_range():
    """Behavioral scores must always be in [0.0, 1.0]."""
    df = load_telemetry("data/telemetry_sample.csv")
    df = _full_pipeline(df)
    scores = df["behavioral_score"]

    assert scores.between(0.0, 1.0).all(), (
        f"Scores out of range: min={scores.min():.4f}, max={scores.max():.4f}"
    )


def test_score_no_nan():
    """No session should have a NaN behavioral score."""
    df = load_telemetry("data/telemetry_sample.csv")
    df = _full_pipeline(df)
    assert not df["behavioral_score"].isna().any(), "NaN behavioral score found."


# ─────────────────────────────────────────────────────────────
# TEST 9: Trend detection — correct labels for known patterns
# ─────────────────────────────────────────────────────────────

def test_trend_u002_improving():
    """U002 is designed with an improving trend. Trend should be 'Improving'."""
    df = load_telemetry("data/telemetry_sample.csv")
    df = _full_pipeline(df)
    u2_trend = df[df["user_id"] == "U002"]["trend"].iloc[-1]
    assert u2_trend == "Improving", f"Expected 'Improving' for U002, got '{u2_trend}'"


def test_trend_u003_changing():
    """U003 is designed with a worsening trend. Trend should be 'Changing'."""
    df = load_telemetry("data/telemetry_sample.csv")
    df = _full_pipeline(df)
    u3_trend = df[df["user_id"] == "U003"]["trend"].iloc[-1]
    assert u3_trend == "Changing", f"Expected 'Changing' for U003, got '{u3_trend}'"


def test_trend_u005_insufficient():
    """U005 has only 2 sessions. Trend should be 'Insufficient Data'."""
    df = load_telemetry("data/telemetry_sample.csv")
    df = _full_pipeline(df)
    u5_trend = df[df["user_id"] == "U005"]["trend"].iloc[-1]
    assert u5_trend == "Insufficient Data", (
        f"Expected 'Insufficient Data' for U005, got '{u5_trend}'"
    )


# ─────────────────────────────────────────────────────────────
# TEST 10: JSON output — valid structure, required keys present
# ─────────────────────────────────────────────────────────────

def test_json_output_structure():
    """Every session in results.json must have the required keys."""
    required_keys = [
        "session_id", "user_id", "timestamp", "metrics",
        "baseline_available", "behavioral_score", "indicator",
        "trend", "top_contributing_factors", "explanation",
        "disclaimer",
    ]
    df = load_telemetry("data/telemetry_sample.csv")
    df = _full_pipeline(df)
    results = build_results(df)

    for r in results:
        for key in required_keys:
            assert key in r, f"Missing key '{key}' in result for session {r.get('session_id')}"


def test_json_scores_valid():
    """behavioral_score in JSON must be a float between 0 and 1."""
    df = load_telemetry("data/telemetry_sample.csv")
    df = _full_pipeline(df)
    results = build_results(df)

    for r in results:
        score = r["behavioral_score"]
        assert isinstance(score, float), f"Score is not float: {type(score)}"
        assert 0.0 <= score <= 1.0, f"Score out of range: {score} in {r['session_id']}"


def test_json_disclaimer_present():
    """Every result must include the PROTOTYPE disclaimer."""
    df = load_telemetry("data/telemetry_sample.csv")
    df = _full_pipeline(df)
    results = build_results(df)

    for r in results:
        assert "disclaimer" in r, "Disclaimer missing from result."
        assert "PROTOTYPE" in r["disclaimer"], "Disclaimer must contain 'PROTOTYPE'."


# ─────────────────────────────────────────────────────────────
# RUN WITHOUT PYTEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_valid_data_loads,
        test_valid_data_full_pipeline,
        test_missing_value_removal,
        test_invalid_numeric_removal,
        test_duplicate_removal,
        test_range_violation_removal,
        test_baseline_calculation_correct,
        test_insufficient_history,
        test_score_range,
        test_score_no_nan,
        test_trend_u002_improving,
        test_trend_u003_changing,
        test_trend_u005_insufficient,
        test_json_output_structure,
        test_json_scores_valid,
        test_json_disclaimer_present,
    ]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}")
            print(f"        {e}")
            failed += 1
    print(f"\n  Results: {passed} passed, {failed} failed out of {len(tests)} tests.")
