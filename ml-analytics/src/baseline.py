"""
baseline.py — Personal Behavioral Baseline computation.

Pipeline step: Preprocessed DataFrame → Per-session baseline comparison

This is the KEY differentiator of this ML module:
  Instead of comparing a user against a population average,
  we compare each session against that user's OWN previous sessions.

Logic:
  For session N of user U:
    baseline = mean of sessions 1 … (N-1)
    If N < MIN_BASELINE_SESSIONS: baseline_available = False

Output per session:
  - baseline_available   (bool)
  - baseline_<feature>   (mean of user's previous sessions)
  - delta_<feature>      (current - baseline; positive = worse)
  - delta_<feature>_pct  (% change from baseline)
  - explanation          (human-readable string)

Project: AI-Based Cognitive Gaming and Memory Assistance Platform
SIH 2026 | Problem ID: SIH26003 | Team ID: 271
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MIN_BASELINE_SESSIONS

# Features used in baseline calculation (normalized forms)
BASELINE_FEATURES = [
    "touch_drift",
    "response_latency",
    "decision_speed",
    "mistake_rate",
    "engagement_score",
]


def compute_personal_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each session, compute the user's personal behavioral baseline
    from all PREVIOUS sessions (not including the current one).

    Returns the input DataFrame with added columns:
        baseline_available        — True if enough history exists
        baseline_<feature>        — Mean of previous sessions for each feature
        delta_<feature>           — Current value minus baseline
        delta_<feature>_pct       — % change from baseline (positive = got worse)
        baseline_explanation      — Human-readable explanation string
    """
    df = df.copy().sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Initialize result columns
    df["baseline_available"] = False
    for feat in BASELINE_FEATURES:
        df[f"baseline_{feat}"]     = np.nan
        df[f"delta_{feat}"]        = np.nan
        df[f"delta_{feat}_pct"]    = np.nan
    df["baseline_explanation"] = ""

    print(f"[BASELINE] Computing personal baselines for {df['user_id'].nunique()} users...")

    for user_id, user_df in df.groupby("user_id"):
        sessions = user_df.sort_values("timestamp")
        n = len(sessions)

        for i, (row_idx, row) in enumerate(sessions.iterrows()):
            # Need at least MIN_BASELINE_SESSIONS previous sessions
            if i < MIN_BASELINE_SESSIONS:
                df.at[row_idx, "baseline_available"] = False
                df.at[row_idx, "baseline_explanation"] = (
                    f"Insufficient history: {i} previous session(s) available "
                    f"(minimum required: {MIN_BASELINE_SESSIONS})."
                )
                continue

            # Baseline = mean of ALL previous sessions for this user
            prev_sessions = sessions.iloc[:i]
            df.at[row_idx, "baseline_available"] = True

            for feat in BASELINE_FEATURES:
                if feat not in sessions.columns:
                    continue
                baseline_val = prev_sessions[feat].mean()
                current_val  = row[feat]
                delta        = current_val - baseline_val
                delta_pct    = (delta / baseline_val * 100) if baseline_val != 0 else 0.0

                df.at[row_idx, f"baseline_{feat}"]    = round(baseline_val, 4)
                df.at[row_idx, f"delta_{feat}"]        = round(delta, 4)
                df.at[row_idx, f"delta_{feat}_pct"]    = round(delta_pct, 2)

            # ── Build explanation string ──
            changes = []
            for feat in BASELINE_FEATURES:
                pct = df.at[row_idx, f"delta_{feat}_pct"]
                if abs(pct) >= 10:   # Only mention notable changes (≥10%)
                    direction = "increased" if pct > 0 else "decreased"
                    label = feat.replace("_", " ")
                    # For engagement, "increased" = better; for others, "increased" = worse
                    if feat == "engagement_score":
                        quality = "improved" if pct > 0 else "declined"
                        changes.append(f"{label} {quality} by {abs(pct):.1f}%")
                    else:
                        changes.append(f"{label} {direction} by {abs(pct):.1f}% vs personal baseline")

            if changes:
                explanation = (
                    f"Compared to personal baseline ({i} previous sessions): "
                    + "; ".join(changes) + "."
                )
            else:
                explanation = (
                    f"Session within expected range of personal baseline "
                    f"({i} previous sessions). No notable change detected."
                )

            df.at[row_idx, "baseline_explanation"] = explanation

    # Summary
    available = df["baseline_available"].sum()
    print(f"[BASELINE] Sessions with baseline available: {available} / {len(df)}")
    print(f"[BASELINE] Sessions without sufficient history: {len(df) - available}")
    return df


def summarize_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Print and return a per-user summary of their baseline statistics.
    Useful for debugging and judge demonstrations.
    """
    rows = []
    for user_id, udf in df.groupby("user_id"):
        base_df = udf[udf["baseline_available"]]
        row = {"user_id": user_id, "total_sessions": len(udf),
               "sessions_with_baseline": len(base_df)}
        for feat in BASELINE_FEATURES:
            col = f"delta_{feat}_pct"
            if col in base_df.columns and len(base_df) > 0:
                row[f"avg_delta_{feat}_pct"] = round(base_df[col].mean(), 2)
        rows.append(row)
    summary = pd.DataFrame(rows)
    return summary


# ─────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.preprocessing import load_telemetry, validate, preprocess, engineer_features
    from src.config import DATA_PATH

    df_raw          = load_telemetry(DATA_PATH)
    df_valid, _     = validate(df_raw)
    df_clean        = preprocess(df_valid)
    df_feat         = engineer_features(df_clean)
    df_base         = compute_personal_baseline(df_feat)

    print("\n=== Baseline summary per user ===")
    print(summarize_baseline(df_base).to_string(index=False))

    print("\n=== Sample explanations ===")
    for _, row in df_base[["session_id", "user_id", "baseline_available", "baseline_explanation"]].iterrows():
        print(f"  [{row['user_id']}] {row['session_id']} | "
              f"baseline={'YES' if row['baseline_available'] else 'NO'} | "
              f"{row['baseline_explanation'][:90]}...")
