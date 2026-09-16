"""
preprocessing.py — Data validation, cleaning, and preprocessing.

Pipeline step: Raw CSV → Validated & Cleaned DataFrame

Project: AI-Based Cognitive Gaming and Memory Assistance Platform
SIH 2026 | Problem ID: SIH26003 | Team ID: 271
"""

import sys
import os
import pandas as pd
import numpy as np

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import REQUIRED_COLUMNS, VALID_RANGES


# ─────────────────────────────────────────────────────────────
# SECTION 1: LOADING
# ─────────────────────────────────────────────────────────────

def load_telemetry(path: str) -> pd.DataFrame:
    """Load telemetry CSV from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Telemetry file not found: {path}")
    df = pd.read_csv(path)
    print(f"[LOAD] Loaded {len(df)} rows from '{path}'")
    return df


# ─────────────────────────────────────────────────────────────
# SECTION 2: VALIDATION
# ─────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Run all validation checks.

    Returns:
        df_valid  — DataFrame with invalid rows removed
        issues    — List of warning/error strings for logging

    Checks:
        1. Required columns present
        2. Duplicate session_ids
        3. Missing values in required fields
        4. Numeric type conversion
        5. Out-of-range values (per config.VALID_RANGES)
        6. Invalid timestamps
    """
    issues = []
    original_count = len(df)

    # ── 1. Required columns ──
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"[VALIDATION ERROR] Missing required columns: {missing_cols}")
    print(f"[VALIDATE] Required columns: OK")

    # ── 2. Duplicate session_ids ──
    dupes = df.duplicated(subset=["session_id"], keep="first")
    if dupes.any():
        n = dupes.sum()
        issues.append(f"Removed {n} duplicate session_id(s).")
        print(f"[VALIDATE] Duplicates: found {n} — removed.")
        df = df[~dupes].copy()
    else:
        print(f"[VALIDATE] Duplicates: none found.")

    # ── 3. Missing values ──
    for col in REQUIRED_COLUMNS:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            issues.append(f"Column '{col}' has {n_missing} missing value(s) — rows removed.")
            print(f"[VALIDATE] Missing in '{col}': {n_missing} rows removed.")
            df = df[df[col].notna()].copy()

    # ── 4. Numeric conversion ──
    numeric_cols = [c for c in VALID_RANGES.keys()]
    for col in numeric_cols:
        if col in df.columns:
            before = len(df)
            df[col] = pd.to_numeric(df[col], errors="coerce")
            coerced = df[col].isna().sum()
            if coerced > 0:
                issues.append(f"Column '{col}' had {coerced} non-numeric value(s) — rows removed.")
                print(f"[VALIDATE] Non-numeric in '{col}': {coerced} rows removed.")
                df = df[df[col].notna()].copy()
    print(f"[VALIDATE] Numeric conversion: OK")

    # ── 5. Range checks ──
    for col, (lo, hi) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        out_of_range = ~df[col].between(lo, hi)
        n = out_of_range.sum()
        if n > 0:
            issues.append(
                f"Column '{col}': {n} value(s) outside [{lo}, {hi}] — rows removed."
            )
            print(f"[VALIDATE] Out-of-range in '{col}': {n} rows removed.")
            df = df[~out_of_range].copy()
    print(f"[VALIDATE] Range checks: OK")

    # ── 6. Timestamp parsing ──
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
        print(f"[VALIDATE] Timestamps: OK")
    except Exception as e:
        issues.append(f"Timestamp parse error: {e}")
        print(f"[VALIDATE] WARNING: Timestamp parse issue — {e}")

    removed = original_count - len(df)
    if removed > 0:
        issues.append(f"Total rows removed by validation: {removed} of {original_count}.")
        print(f"[VALIDATE] Summary: {removed} rows removed, {len(df)} rows remain.")
    else:
        print(f"[VALIDATE] Summary: All {len(df)} rows passed validation.")

    return df, issues


# ─────────────────────────────────────────────────────────────
# SECTION 3: PREPROCESSING
# ─────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare data after validation.

    Steps:
        1. Sort each user's sessions chronologically
        2. Compute session index per user (1 = oldest, N = newest)
        3. Compute mistake_rate = mistakes / (estimated decisions)
        4. Clip and round numeric columns to clean values

    Returns a clean, preprocessed DataFrame ready for feature engineering.
    """
    df = df.copy()

    # ── 1. Sort chronologically within each user ──
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    print(f"[PREPROCESS] Sorted by user_id + timestamp.")

    # ── 2. Session index per user ──
    df["session_index"] = df.groupby("user_id").cumcount() + 1
    print(f"[PREPROCESS] Session index computed.")

    # ── 3. Mistake rate ──
    # Estimate decisions from session_duration / avg decision_speed (floored at 1)
    df["estimated_decisions"] = np.maximum(
        (df["session_duration"] / df["decision_speed"]).round().astype(int), 1
    )
    df["mistake_rate"] = (df["mistakes"] / df["estimated_decisions"]).clip(0, 1).round(4)
    print(f"[PREPROCESS] Mistake rate computed.")

    # ── 4. Engagement score (proxy: completion_rate × session_duration / max_session_duration) ──
    max_dur = df["session_duration"].max()
    df["engagement_score"] = (
        df["completion_rate"] * (df["session_duration"] / max_dur)
    ).clip(0, 1).round(4)
    print(f"[PREPROCESS] Engagement score computed.")

    # ── 5. Round numeric columns ──
    for col in ["touch_drift", "response_latency", "decision_speed", "session_duration"]:
        df[col] = df[col].round(2)

    print(f"[PREPROCESS] Complete. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────
# SECTION 4: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create normalized behavioral features for scoring and baseline comparison.

    For each numeric feature, compute a user-level min-max normalization
    so that values are comparable across different users.

    Normalized scale: 0.0 (best observed for that user) → 1.0 (worst observed)

    Features normalized:
        - touch_drift_norm
        - response_latency_norm
        - decision_speed_norm
        - mistake_rate_norm
        - engagement_score_norm  (inverted: higher engagement = lower norm value)
    """
    df = df.copy()
    print(f"[FEATURES] Starting feature engineering...")

    def user_minmax(series: pd.Series, group_keys: pd.Series, invert: bool = False) -> pd.Series:
        """
        Normalize within each user group.
        invert=True means higher raw value = better (e.g. completion_rate).
        """
        result = pd.Series(index=series.index, dtype=float)
        for user_id, grp_idx in group_keys.groupby(group_keys).groups.items():
            vals = series.loc[grp_idx]
            lo, hi = vals.min(), vals.max()
            if hi == lo:
                norm = pd.Series(0.5, index=grp_idx)
            else:
                norm = (vals - lo) / (hi - lo)
            if invert:
                norm = 1.0 - norm
            result.loc[grp_idx] = norm
        return result.round(4)

    df["touch_drift_norm"]        = user_minmax(df["touch_drift"],        df["user_id"])
    df["response_latency_norm"]   = user_minmax(df["response_latency"],   df["user_id"])
    df["decision_speed_norm"]     = user_minmax(df["decision_speed"],     df["user_id"])
    df["mistake_rate_norm"]       = user_minmax(df["mistake_rate"],       df["user_id"])
    df["engagement_score_norm"]   = user_minmax(df["engagement_score"],   df["user_id"], invert=True)

    print(f"[FEATURES] Normalized features: touch_drift_norm, response_latency_norm, "
          f"decision_speed_norm, mistake_rate_norm, engagement_score_norm")
    print(f"[FEATURES] Feature engineering complete.")
    return df


# ─────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.config import DATA_PATH
    df_raw   = load_telemetry(DATA_PATH)
    df_valid, issues = validate(df_raw)
    df_clean = preprocess(df_valid)
    df_feat  = engineer_features(df_clean)
    print("\n=== Sample output (first 3 rows, key columns) ===")
    cols = ["session_id", "user_id", "session_index", "touch_drift_norm",
            "response_latency_norm", "mistake_rate_norm", "engagement_score_norm"]
    print(df_feat[cols].head(3).to_string(index=False))
    if issues:
        print("\n=== Validation Issues ===")
        for i in issues:
            print(" •", i)
