"""
main.py — ML & Predictive Analytics Pipeline — Main Entrypoint

Usage:
    python3 main.py                    # Run full pipeline
    python3 main.py --user U001        # Analyze one user only
    python3 main.py --data path.csv    # Use a different data file

Pipeline:
    Load → Validate → Preprocess → Feature Engineering →
    Personal Baseline → Behavioral Scoring → Trend Detection →
    Visualization → JSON/CSV Output

Outputs:
    outputs/results.json   — Full JSON per session (for dashboard)
    outputs/scores.csv     — Scores table (for quick review)
    outputs/charts/        — 5 visualization charts

DISCLAIMER:
    PROTOTYPE ONLY. Not a medical diagnostic system.
    Analyzes gameplay behavior only. Does not diagnose dementia.

Project: AI-Based Cognitive Gaming and Memory Assistance Platform
SIH 2026 | Problem ID: SIH26003 | Team ID: 271
"""

import sys
import argparse
import os

# ── Make src importable when run from project root ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config       import DATA_PATH, DISCLAIMER
from src.preprocessing import load_telemetry, validate, preprocess, engineer_features
from src.baseline      import compute_personal_baseline
from src.scoring       import compute_scores
from src.analytics     import detect_trends, build_results, save_outputs, generate_all_charts


def print_header():
    print()
    print("=" * 72)
    print("  ML & Predictive Analytics Pipeline")
    print("  AI-Based Cognitive Gaming Platform — SIH 2026 | SIH26003")
    print("=" * 72)
    print(f"  {DISCLAIMER[:80]}...")
    print("=" * 72)
    print()


def print_summary(df, results):
    """Print a clean per-user summary to the console."""
    print()
    print("─" * 72)
    print("  RESULTS SUMMARY")
    print("─" * 72)

    for user_id, udf in df.groupby("user_id"):
        latest = udf.sort_values("timestamp").iloc[-1]
        n_sessions = len(udf)
        trend = latest["trend"]
        score = latest["behavioral_score"]
        indicator = latest["indicator"]
        baseline_ok = latest["baseline_available"]

        print(f"\n  User: {user_id}  |  Sessions analysed: {n_sessions}")
        print(f"  Trend            : {trend}")
        print(f"  Latest Score     : {score:.3f}  (0=best, 1=most changed)")
        print(f"  Indicator        : {indicator}")
        print(f"  Baseline         : {'Available' if baseline_ok else 'Insufficient history'}")

        if baseline_ok:
            top = latest.get("top_factors", "")
            if top:
                print(f"  Top Factors      : {top[:80]}")

    print()
    print("─" * 72)
    print(f"  Outputs written to:")
    print(f"    outputs/results.json")
    print(f"    outputs/scores.csv")
    print(f"    outputs/charts/  (5 charts)")
    print("─" * 72)
    print()


def run_pipeline(data_path: str, filter_user: str = None):
    """Execute the full analytics pipeline."""
    print_header()

    # ── STEP 1: Load ──
    print("[STEP 1/7] Loading telemetry data...")
    df = load_telemetry(data_path)

    # ── STEP 2: Validate ──
    print("\n[STEP 2/7] Validating data...")
    df, issues = validate(df)
    if issues:
        print("  Issues found during validation:")
        for i in issues:
            print(f"    • {i}")

    # ── STEP 3: Preprocess ──
    print("\n[STEP 3/7] Preprocessing...")
    df = preprocess(df)

    # ── STEP 4: Feature Engineering ──
    print("\n[STEP 4/7] Engineering features...")
    df = engineer_features(df)

    # Optional: filter to one user for focused analysis
    if filter_user:
        if filter_user not in df["user_id"].values:
            print(f"[ERROR] User '{filter_user}' not found in dataset.")
            sys.exit(1)
        df = df[df["user_id"] == filter_user].copy()
        print(f"  Filtered to user: {filter_user} ({len(df)} sessions)")

    # ── STEP 5: Personal Baseline ──
    print("\n[STEP 5/7] Computing personal behavioral baselines...")
    df = compute_personal_baseline(df)

    # ── STEP 6: Behavioral Scoring ──
    print("\n[STEP 6/7] Computing behavioral scores...")
    df = compute_scores(df)

    # ── STEP 7: Trend Detection ──
    print("\n[STEP 7/7] Detecting behavioral trends...")
    df = detect_trends(df)

    # ── Save Outputs ──
    print("\n[OUTPUT] Saving JSON and CSV...")
    results = build_results(df)
    save_outputs(df, results)

    # ── Charts ──
    print("\n[CHARTS] Generating visualizations...")
    generate_all_charts(df)

    # ── Console Summary ──
    print_summary(df, results)


def main():
    parser = argparse.ArgumentParser(
        description="ML & Predictive Analytics Pipeline — SIH 2026"
    )
    parser.add_argument(
        "--data", type=str, default=DATA_PATH,
        help=f"Path to telemetry CSV (default: {DATA_PATH})"
    )
    parser.add_argument(
        "--user", type=str, default=None,
        help="Analyze a single user only (e.g. --user U001)"
    )
    args = parser.parse_args()
    run_pipeline(data_path=args.data, filter_user=args.user)


if __name__ == "__main__":
    main()
