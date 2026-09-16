"""
analytics.py — Trend Detection, Explainable Output, and Visualization.

Pipeline steps:
  1. Trend detection — compare recent vs. earlier session scores
  2. Full explainable JSON/CSV output generation
  3. Chart generation — 5 chart types saved to outputs/charts/

DISCLAIMER: Prototype only. Not a clinical diagnostic system.

Project: AI-Based Cognitive Gaming and Memory Assistance Platform
SIH 2026 | Problem ID: SIH26003 | Team ID: 271
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # No display — save to file only
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    TREND_RECENT_WINDOW,
    TREND_IMPROVING_THRESHOLD,
    TREND_CHANGING_THRESHOLD,
    OUTPUT_JSON,
    OUTPUT_CSV,
    CHARTS_DIR,
    DISCLAIMER,
)

CHART_STYLE = {
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor":   "#16213e",
    "axes.edgecolor":   "#0f3460",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#e0e0e0",
    "ytick.color":      "#e0e0e0",
    "text.color":       "#e0e0e0",
    "grid.color":       "#0f3460",
    "grid.alpha":       0.5,
    "lines.linewidth":  2.2,
}
plt.rcParams.update(CHART_STYLE)

COLORS = {
    "U001": "#4cc9f0",
    "U002": "#4ade80",
    "U003": "#f72585",
    "U004": "#ffd166",
    "U005": "#b5838d",
}
DEFAULT_COLOR = "#a8dadc"


# ─────────────────────────────────────────────────────────────
# TREND DETECTION
# ─────────────────────────────────────────────────────────────

def detect_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each user, compare their most recent N session scores
    to all earlier session scores.

    Returns a new column 'trend' per session (applied to all sessions
    of that user — the trend describes the user's overall trajectory).

    Values:
        Improving  — recent avg score is notably lower (better)
        Stable     — recent and earlier scores are similar
        Changing   — recent avg score is notably higher (worse)
        Insufficient Data — fewer than TREND_RECENT_WINDOW + 1 sessions
    """
    df = df.copy()
    df["trend"] = "Insufficient Data"

    print(f"[TREND] Detecting trends (recent window = {TREND_RECENT_WINDOW} sessions)...")

    for user_id, udf in df.groupby("user_id"):
        udf_sorted = udf.sort_values("timestamp")
        n = len(udf_sorted)

        if n <= TREND_RECENT_WINDOW:
            # Not enough sessions to split into earlier + recent
            df.loc[udf_sorted.index, "trend"] = "Insufficient Data"
            continue

        recent_idx  = udf_sorted.index[-TREND_RECENT_WINDOW:]
        earlier_idx = udf_sorted.index[:-TREND_RECENT_WINDOW]

        recent_avg  = df.loc[recent_idx, "behavioral_score"].mean()
        earlier_avg = df.loc[earlier_idx, "behavioral_score"].mean()
        diff        = recent_avg - earlier_avg  # negative = improved (lower score = better)

        if diff <= TREND_IMPROVING_THRESHOLD:
            trend = "Improving"
        elif diff >= TREND_CHANGING_THRESHOLD:
            trend = "Changing"
        else:
            trend = "Stable"

        df.loc[udf_sorted.index, "trend"] = trend
        print(f"  [{user_id}] earlier_avg={earlier_avg:.3f} | recent_avg={recent_avg:.3f} "
              f"| diff={diff:+.3f} → {trend}")

    return df


# ─────────────────────────────────────────────────────────────
# JSON / CSV OUTPUT
# ─────────────────────────────────────────────────────────────

def build_results(df: pd.DataFrame) -> list[dict]:
    """
    Build a list of result dicts — one per session.
    Each dict is structured for dashboard consumption.
    """
    results = []

    for _, row in df.iterrows():
        factors_raw = str(row.get("top_factors", ""))
        factors_list = [f.strip() for f in factors_raw.split("|") if f.strip()]

        entry = {
            "session_id":          row["session_id"],
            "user_id":             row["user_id"],
            "timestamp":           str(row["timestamp"]),
            "game_type":           str(row.get("game_type", "unknown")),
            # ── Raw metrics ──
            "metrics": {
                "touch_drift":       float(row["touch_drift"]),
                "response_latency":  float(row["response_latency"]),
                "decision_speed":    float(row["decision_speed"]),
                "mistakes":          int(row["mistakes"]),
                "mistake_rate":      float(row["mistake_rate"]),
                "completion_rate":   float(row["completion_rate"]),
                "session_duration":  float(row["session_duration"]),
                "engagement_score":  float(row["engagement_score"]),
            },
            # ── Baseline ──
            "baseline_available": bool(row["baseline_available"]),
            "baseline": {
                "touch_drift":      _safe_float(row, "baseline_touch_drift"),
                "response_latency": _safe_float(row, "baseline_response_latency"),
                "decision_speed":   _safe_float(row, "baseline_decision_speed"),
                "mistake_rate":     _safe_float(row, "baseline_mistake_rate"),
                "engagement_score": _safe_float(row, "baseline_engagement_score"),
            } if row["baseline_available"] else None,
            "baseline_deltas_pct": {
                "touch_drift":      _safe_float(row, "delta_touch_drift_pct"),
                "response_latency": _safe_float(row, "delta_response_latency_pct"),
                "decision_speed":   _safe_float(row, "delta_decision_speed_pct"),
                "mistake_rate":     _safe_float(row, "delta_mistake_rate_pct"),
                "engagement_score": _safe_float(row, "delta_engagement_score_pct"),
            } if row["baseline_available"] else None,
            # ── Score + Output ──
            "behavioral_score":    float(row["behavioral_score"]),
            "score_method":        str(row["score_method"]),
            "indicator":           str(row["indicator"]),
            "trend":               str(row["trend"]),
            "top_contributing_factors": factors_list,
            "explanation":         str(row.get("baseline_explanation", "")),
            # ── Metadata ──
            "data_note":     "SYNTHETIC — NOT clinical data",
            "disclaimer":    DISCLAIMER,
            "generated_at":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        results.append(entry)

    return results


def _safe_float(row, col):
    val = row.get(col, None)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(float(val), 4)


def save_outputs(df: pd.DataFrame, results: list[dict]):
    """Save results.json and scores.csv."""
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    # JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump({"disclaimer": DISCLAIMER, "sessions": results}, f, indent=2)
    print(f"[OUTPUT] JSON saved → {OUTPUT_JSON}")

    # CSV (key columns only for dashboard quick-load)
    score_cols = [
        "session_id", "user_id", "timestamp", "game_type",
        "touch_drift", "response_latency", "decision_speed",
        "mistakes", "mistake_rate", "completion_rate", "session_duration",
        "engagement_score", "behavioral_score", "indicator", "trend",
        "baseline_available", "score_method", "top_factors",
    ]
    available_cols = [c for c in score_cols if c in df.columns]
    df[available_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"[OUTPUT] CSV  saved → {OUTPUT_CSV}")


# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────

def _savefig(name: str):
    path = os.path.join(CHARTS_DIR, name)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[CHART] Saved → {path}")


def chart_behavioral_scores(df: pd.DataFrame):
    """Chart 1: Behavioral score across sessions per user."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for user_id, udf in df.groupby("user_id"):
        udf_s = udf.sort_values("timestamp")
        color = COLORS.get(user_id, DEFAULT_COLOR)
        trend = udf_s["trend"].iloc[-1]
        ax.plot(
            udf_s["session_index"], udf_s["behavioral_score"],
            marker="o", color=color, label=f"{user_id} ({trend})", markersize=5,
        )

    ax.axhline(0.70, color="#f72585", linestyle="--", alpha=0.7, label="Notable change threshold (0.70)")
    ax.axhline(0.45, color="#ffd166", linestyle="--", alpha=0.7, label="Moderate variation threshold (0.45)")
    ax.set_xlabel("Session Index")
    ax.set_ylabel("Behavioral Score (0=best, 1=worst)")
    ax.set_title("Behavioral Score Across Sessions — All Users", fontsize=13, pad=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, linestyle="--")
    ax.set_ylim(0, 1.05)
    _savefig("01_behavioral_scores.png")


def chart_personal_baseline_comparison(df: pd.DataFrame):
    """Chart 2: Current vs baseline — one subplot per user."""
    features = ["touch_drift", "response_latency", "decision_speed", "mistake_rate"]
    users_with_baseline = df[df["baseline_available"]]["user_id"].unique()

    if len(users_with_baseline) == 0:
        print("[CHART] No baseline data — skipping comparison chart.")
        return

    n_users = len(users_with_baseline)
    fig, axes = plt.subplots(1, n_users, figsize=(5 * n_users, 5), sharey=False)
    if n_users == 1:
        axes = [axes]

    for ax, user_id in zip(axes, users_with_baseline):
        udf = df[(df["user_id"] == user_id) & (df["baseline_available"])].sort_values("timestamp")
        last = udf.iloc[-1]
        color = COLORS.get(user_id, DEFAULT_COLOR)

        current_vals  = [last[f] for f in features]
        baseline_vals = [last[f"baseline_{f}"] for f in features]
        labels        = ["Touch Drift", "Resp. Latency", "Decis. Speed", "Mistake Rate"]

        x = np.arange(len(features))
        w = 0.35
        ax.bar(x - w/2, baseline_vals, w, label="Personal Baseline", color="#4361ee", alpha=0.85)
        ax.bar(x + w/2, current_vals,  w, label="Latest Session",    color=color, alpha=0.85)
        ax.set_title(f"{user_id}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Latest Session vs. Personal Baseline — Per User", fontsize=13, y=1.02)
    plt.tight_layout()
    _savefig("02_baseline_comparison.png")


def chart_touch_drift_trend(df: pd.DataFrame):
    """Chart 3: Touch drift over sessions per user."""
    fig, ax = plt.subplots(figsize=(11, 4))
    for user_id, udf in df.groupby("user_id"):
        udf_s = udf.sort_values("timestamp")
        color = COLORS.get(user_id, DEFAULT_COLOR)
        ax.plot(udf_s["session_index"], udf_s["touch_drift"],
                marker="s", color=color, label=user_id, markersize=4)
    ax.set_xlabel("Session Index")
    ax.set_ylabel("Touch Drift (pixels)")
    ax.set_title("Touch Drift Over Sessions", fontsize=13, pad=12)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--")
    _savefig("03_touch_drift_trend.png")


def chart_response_latency_trend(df: pd.DataFrame):
    """Chart 4: Response latency over sessions per user."""
    fig, ax = plt.subplots(figsize=(11, 4))
    for user_id, udf in df.groupby("user_id"):
        udf_s = udf.sort_values("timestamp")
        color = COLORS.get(user_id, DEFAULT_COLOR)
        ax.plot(udf_s["session_index"], udf_s["response_latency"],
                marker="D", color=color, label=user_id, markersize=4)
    ax.set_xlabel("Session Index")
    ax.set_ylabel("Response Latency (ms)")
    ax.set_title("Response Latency Over Sessions", fontsize=13, pad=12)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--")
    _savefig("04_response_latency_trend.png")


def chart_decision_speed_trend(df: pd.DataFrame):
    """Chart 5: Decision speed over sessions per user."""
    fig, ax = plt.subplots(figsize=(11, 4))
    for user_id, udf in df.groupby("user_id"):
        udf_s = udf.sort_values("timestamp")
        color = COLORS.get(user_id, DEFAULT_COLOR)
        ax.plot(udf_s["session_index"], udf_s["decision_speed"],
                marker="^", color=color, label=user_id, markersize=4)
    ax.set_xlabel("Session Index")
    ax.set_ylabel("Decision Speed (seconds/decision)")
    ax.set_title("Decision Speed Over Sessions", fontsize=13, pad=12)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--")
    _savefig("05_decision_speed_trend.png")


def generate_all_charts(df: pd.DataFrame):
    """Run all 5 chart generation functions."""
    print(f"[CHARTS] Generating charts → {CHARTS_DIR}/")
    chart_behavioral_scores(df)
    chart_personal_baseline_comparison(df)
    chart_touch_drift_trend(df)
    chart_response_latency_trend(df)
    chart_decision_speed_trend(df)
    print(f"[CHARTS] All charts saved.")


# ─────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.preprocessing import load_telemetry, validate, preprocess, engineer_features
    from src.baseline import compute_personal_baseline
    from src.scoring import compute_scores
    from src.config import DATA_PATH

    df = load_telemetry(DATA_PATH)
    df, _ = validate(df)
    df = preprocess(df)
    df = engineer_features(df)
    df = compute_personal_baseline(df)
    df = compute_scores(df)
    df = detect_trends(df)

    print("\n=== Trend Summary ===")
    print(df.groupby(["user_id", "trend"])["session_id"].count().to_string())

    results = build_results(df)
    save_outputs(df, results)
    generate_all_charts(df)

    print("\n=== Sample JSON result (last session of U003) ===")
    u3 = [r for r in results if r["user_id"] == "U003"][-1]
    print(json.dumps(u3, indent=2))
