"""
scoring.py — Behavioral Scoring Engine.

Pipeline step: Baseline DataFrame → Behavioral Score + Indicator per session

FORMULA:
  For sessions WITH a baseline:
    weighted_change = Σ weight_i × clipped_delta_pct_i   (for each feature)
    behavioral_score = sigmoid(weighted_change / 100)     (maps to 0–1)

  For sessions WITHOUT a baseline:
    raw_score = Σ weight_i × normalized_feature_i
    behavioral_score = raw_score                          (0–1, direct composite)

  Score interpretation:
    0.0 → 0.45   Within expected range
    0.45 → 0.70  Moderate behavioral variation
    0.70 → 1.0   Notable behavioral change detected

IMPORTANT:
  This is a PROTOTYPE BEHAVIORAL ANALYTICS SCORE.
  It is NOT a clinical measure.
  It does NOT diagnose any medical condition.

Project: AI-Based Cognitive Gaming and Memory Assistance Platform
SIH 2026 | Problem ID: SIH26003 | Team ID: 271
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    SCORE_WEIGHTS,
    INDICATOR_HIGH_CHANGE,
    INDICATOR_MODERATE,
    DISCLAIMER,
)

# Map config feature names to delta_pct column names
FEATURE_TO_DELTA = {
    "touch_drift":      "delta_touch_drift_pct",
    "response_latency": "delta_response_latency_pct",
    "decision_speed":   "delta_decision_speed_pct",
    "mistake_rate":     "delta_mistake_rate_pct",
    "completion_rate":  "delta_engagement_score_pct",    # mapped to engagement
}

# Map config feature names to normalized column names (fallback when no baseline)
FEATURE_TO_NORM = {
    "touch_drift":      "touch_drift_norm",
    "response_latency": "response_latency_norm",
    "decision_speed":   "decision_speed_norm",
    "mistake_rate":     "mistake_rate_norm",
    "completion_rate":  "engagement_score_norm",
}


def _sigmoid(x: float) -> float:
    """Map any real number to (0, 1). Used to compress the weighted change."""
    return 1.0 / (1.0 + np.exp(-x))


def _compute_score_with_baseline(row: pd.Series) -> float:
    """
    Score for sessions that have a personal baseline.
    Uses weighted delta (% change from baseline).

    Positive delta = feature got worse → score increases toward 1.0
    For engagement_score, positive delta = better → negated before weighting.
    """
    weighted_sum = 0.0
    for feat, weight in SCORE_WEIGHTS.items():
        col = FEATURE_TO_DELTA.get(feat)
        if col and col in row.index and not pd.isna(row[col]):
            delta_pct = row[col]
            # Invert engagement: higher engagement is BETTER, so decline → worse
            if feat == "completion_rate":
                delta_pct = -delta_pct
            # Clip extreme outliers to ±100%
            delta_pct = np.clip(delta_pct, -100, 100)
            weighted_sum += weight * delta_pct

    # Sigmoid maps weighted_sum (typically -100 to +100) to (0, 1)
    # We scale by 0.05 so ±20% change maps to roughly 0.37–0.63
    score = _sigmoid(weighted_sum * 0.05)
    return round(float(score), 4)


def _compute_score_without_baseline(row: pd.Series) -> float:
    """
    Fallback score for sessions with insufficient history.
    Uses normalized feature values directly (0=best, 1=worst observed for this user).
    """
    weighted_sum = 0.0
    for feat, weight in SCORE_WEIGHTS.items():
        col = FEATURE_TO_NORM.get(feat)
        if col and col in row.index and not pd.isna(row[col]):
            val = row[col]
            # Invert engagement: lower norm = better
            if feat == "completion_rate":
                val = 1.0 - val
            weighted_sum += weight * val

    return round(float(np.clip(weighted_sum, 0.0, 1.0)), 4)


def _get_indicator(score: float, baseline_available: bool) -> str:
    """Map behavioral score to a plain-language indicator."""
    prefix = "" if baseline_available else "[No baseline — raw score] "
    if score >= INDICATOR_HIGH_CHANGE:
        return prefix + "Notable behavioral change detected"
    elif score >= INDICATOR_MODERATE:
        return prefix + "Moderate behavioral variation detected"
    else:
        return prefix + "Behavioral pattern within expected range"


def _get_top_factors(row: pd.Series, baseline_available: bool, n: int = 3) -> list[str]:
    """
    Return the top N contributing factors as human-readable strings.
    For sessions with a baseline: based on % delta.
    For sessions without: based on normalized values.
    """
    factor_labels = {
        "touch_drift":      "touch drift",
        "response_latency": "response latency",
        "decision_speed":   "decision speed",
        "mistake_rate":     "mistake rate",
        "completion_rate":  "engagement / completion",
    }

    contributions = {}
    if baseline_available:
        for feat in SCORE_WEIGHTS:
            col = FEATURE_TO_DELTA.get(feat)
            if col and col in row.index and not pd.isna(row[col]):
                pct = row[col]
                if feat == "completion_rate":
                    pct = -pct
                contributions[feat] = abs(pct)
    else:
        for feat in SCORE_WEIGHTS:
            col = FEATURE_TO_NORM.get(feat)
            if col and col in row.index and not pd.isna(row[col]):
                contributions[feat] = row[col]

    sorted_feats = sorted(contributions, key=contributions.get, reverse=True)[:n]

    factors = []
    for feat in sorted_feats:
        label = factor_labels.get(feat, feat)
        if baseline_available:
            delta_col = FEATURE_TO_DELTA.get(feat)
            if delta_col and delta_col in row.index:
                pct = row[delta_col]
                if feat == "completion_rate":
                    pct = -pct
                direction = "increased" if pct > 0 else "decreased"
                if feat == "completion_rate":
                    direction = "declined" if pct > 0 else "improved"
                factors.append(f"{label} {direction} by {abs(row[delta_col]):.1f}% vs personal baseline")
        else:
            factors.append(f"{label} (normalized value: {contributions[feat]:.2f})")

    return factors


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute behavioral score and indicator for every session.

    Adds columns:
        behavioral_score     — float 0–1 (lower is better)
        indicator            — plain-language behavioral indicator string
        top_factors          — list of top contributing features (stored as string)
        score_method         — 'baseline_delta' or 'raw_normalized'
    """
    df = df.copy()
    scores, indicators, top_factors_list, methods = [], [], [], []

    print(f"[SCORING] Computing behavioral scores for {len(df)} sessions...")

    for _, row in df.iterrows():
        baseline_avail = bool(row.get("baseline_available", False))

        if baseline_avail:
            score  = _compute_score_with_baseline(row)
            method = "baseline_delta"
        else:
            score  = _compute_score_without_baseline(row)
            method = "raw_normalized"

        indicator = _get_indicator(score, baseline_avail)
        factors   = _get_top_factors(row, baseline_avail)

        scores.append(score)
        indicators.append(indicator)
        top_factors_list.append(" | ".join(factors))
        methods.append(method)

    df["behavioral_score"] = scores
    df["indicator"]        = indicators
    df["top_factors"]      = top_factors_list
    df["score_method"]     = methods

    print(f"[SCORING] Done. Score range: "
          f"{df['behavioral_score'].min():.3f} – {df['behavioral_score'].max():.3f}")
    return df


# ─────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.preprocessing import load_telemetry, validate, preprocess, engineer_features
    from src.baseline import compute_personal_baseline
    from src.config import DATA_PATH

    df = load_telemetry(DATA_PATH)
    df, _ = validate(df)
    df = preprocess(df)
    df = engineer_features(df)
    df = compute_personal_baseline(df)
    df = compute_scores(df)

    print("\n=== Behavioral Score Summary per User ===")
    summary = (
        df.groupby("user_id")["behavioral_score"]
        .agg(["count", "mean", "min", "max"])
        .round(3)
        .rename(columns={"count": "sessions", "mean": "avg_score",
                          "min": "min_score", "max": "max_score"})
    )
    print(summary.to_string())

    print("\n=== Sample Session Results ===")
    cols = ["session_id", "user_id", "behavioral_score", "indicator", "score_method"]
    print(df[cols].to_string(index=False))
