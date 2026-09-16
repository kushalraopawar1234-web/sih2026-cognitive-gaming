"""
config.py — Configurable parameters for the ML & Predictive Analytics pipeline.

All weights, thresholds, and settings live here.
To tune the system, ONLY change values in this file.

Project: AI-Based Cognitive Gaming and Memory Assistance Platform
SIH 2026 | Problem ID: SIH26003 | Team ID: 271
"""

import os

_ML_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DATA_PATH = os.path.join(_ML_DIR, "data", "telemetry_sample.csv")
OUTPUT_JSON = os.path.join(_ML_DIR, "outputs", "results.json")
OUTPUT_CSV = os.path.join(_ML_DIR, "outputs", "scores.csv")
CHARTS_DIR = os.path.join(_ML_DIR, "outputs", "charts")

# ─────────────────────────────────────────────
# TELEMETRY SCHEMA — Required columns
# ─────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "session_id",
    "user_id",
    "timestamp",
    "touch_drift",
    "response_latency",
    "decision_speed",
    "mistakes",
    "completion_rate",
    "session_duration",
]

# ─────────────────────────────────────────────
# VALIDATION RULES
# ─────────────────────────────────────────────
VALID_RANGES = {
    "touch_drift":       (0.0, 500.0),   # pixels
    "response_latency":  (0.0, 30000.0), # milliseconds
    "decision_speed":    (0.0, 60.0),    # seconds per decision
    "mistakes":          (0, 200),       # count
    "completion_rate":   (0.0, 1.0),     # fraction 0-1
    "session_duration":  (1.0, 3600.0),  # seconds
}

# Minimum sessions required before a personal baseline is considered reliable
MIN_BASELINE_SESSIONS = 3

# ─────────────────────────────────────────────
# BEHAVIORAL SCORE WEIGHTS
# (must sum to 1.0)
# ─────────────────────────────────────────────
# Positive weight  = higher value is WORSE (e.g. more mistakes = lower score)
# Negative weight  = handled by inversion inside scoring.py

SCORE_WEIGHTS = {
    "touch_drift":      0.20,   # higher drift → worse
    "response_latency": 0.25,   # higher latency → worse
    "decision_speed":   0.20,   # higher (slower) speed → worse
    "mistake_rate":     0.20,   # higher rate → worse
    "completion_rate":  0.15,   # higher completion → BETTER (inverted inside scorer)
}

# Score range for normalization output
SCORE_MIN = 0.0
SCORE_MAX = 1.0

# ─────────────────────────────────────────────
# TREND DETECTION
# ─────────────────────────────────────────────
# How many of the most recent sessions to compare against baseline
TREND_RECENT_WINDOW = 3

# Thresholds for labelling trend
# If recent avg score differs from baseline avg score by more than this → trend label
TREND_IMPROVING_THRESHOLD = -0.08   # recent score is this much LOWER than baseline (i.e. better)
TREND_CHANGING_THRESHOLD  =  0.08   # recent score is this much HIGHER than baseline (i.e. worse)
# Between these two values → STABLE

# ─────────────────────────────────────────────
# INDICATOR THRESHOLDS
# ─────────────────────────────────────────────
# Behavioral score buckets (prototype — NOT clinical)
INDICATOR_HIGH_CHANGE   = 0.70   # score >= 0.70  → "Notable behavioral change detected"
INDICATOR_MODERATE      = 0.45   # score >= 0.45  → "Moderate behavioral variation detected"
INDICATOR_LOW           = 0.00   # score <  0.45  → "Behavioral pattern within expected range"

# ─────────────────────────────────────────────
# DISCLAIMER (printed in all outputs)
# ─────────────────────────────────────────────
DISCLAIMER = (
    "PROTOTYPE ONLY — NOT a medical diagnostic system. "
    "This module analyzes gameplay behavior and highlights behavioral changes "
    "for evaluator support. It does not diagnose dementia or any medical condition. "
    "All values are computed from synthetic demonstration data."
)
