"""
generate_data.py — Generates synthetic telemetry data for pipeline development.

IMPORTANT: This is SYNTHETIC DEMONSTRATION DATA only.
           NOT real patient data. NOT clinically validated values.
           For development and SIH demonstration purposes only.

Generates 5 users × varied sessions to demonstrate:
  U001 — Stable behavior
  U002 — Improving behavior (scores getting better over time)
  U003 — Changing behavior (scores deteriorating over time)
  U004 — Mixed/fluctuating behavior
  U005 — Insufficient history (only 2 sessions — tests baseline fallback)
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)  # Reproducible results

def generate_sessions(
    user_id,
    n_sessions,
    start_date="2026-07-01",
    drift_trend=0.0,       # positive = getting worse over time
    latency_trend=0.0,
    speed_trend=0.0,
    mistake_trend=0.0,
    base_drift=25.0,
    base_latency=1200.0,
    base_speed=3.5,
    base_mistakes=5,
    base_completion=0.85,
    noise_scale=0.12,
):
    """Generate sessions for one user with configurable trends and noise."""
    rows = []
    dates = pd.date_range(start=start_date, periods=n_sessions, freq="2D")

    for i, ts in enumerate(dates):
        progress = i / max(n_sessions - 1, 1)   # 0.0 → 1.0 over all sessions
        noise = lambda base: base * (1 + np.random.normal(0, noise_scale))

        session_duration = np.clip(noise(600), 60, 1800)  # 1 min to 30 min
        mistakes_raw = int(np.clip(
            base_mistakes * (1 + mistake_trend * progress) + np.random.randint(-2, 3),
            0, 50
        ))
        decisions = np.random.randint(15, 35)
        mistake_rate = round(mistakes_raw / decisions, 4)
        completion = round(np.clip(
            base_completion + np.random.normal(0, 0.05), 0.0, 1.0
        ), 4)

        rows.append({
            "session_id":       f"{user_id}_S{i+1:03d}",
            "user_id":          user_id,
            "timestamp":        ts.strftime("%Y-%m-%d %H:%M:%S"),
            "touch_drift":      round(np.clip(noise(base_drift * (1 + drift_trend * progress)), 0, 500), 2),
            "response_latency": round(np.clip(noise(base_latency * (1 + latency_trend * progress)), 100, 20000), 2),
            "decision_speed":   round(np.clip(noise(base_speed * (1 + speed_trend * progress)), 0.1, 60), 2),
            "mistakes":         mistakes_raw,
            "completion_rate":  completion,
            "session_duration": round(session_duration, 2),
            "game_type":        np.random.choice(["memory_match", "pattern_recall", "color_sort"]),
            "data_note":        "SYNTHETIC — NOT clinical data",
        })

    return rows


def main():
    all_rows = []

    # U001 — Stable (small noise, no trend)
    all_rows += generate_sessions(
        "U001", n_sessions=12,
        base_drift=22, base_latency=1100, base_speed=3.2,
        base_mistakes=4, base_completion=0.88,
        drift_trend=0.0, latency_trend=0.0, speed_trend=0.0,
        noise_scale=0.10,
    )

    # U002 — Improving (trends negative = values getting better)
    all_rows += generate_sessions(
        "U002", n_sessions=10,
        base_drift=35, base_latency=1800, base_speed=5.0,
        base_mistakes=8, base_completion=0.72,
        drift_trend=-0.45, latency_trend=-0.40, speed_trend=-0.35,
        mistake_trend=-0.50, noise_scale=0.12,
    )

    # U003 — Changing (trends positive = values getting worse)
    all_rows += generate_sessions(
        "U003", n_sessions=11,
        base_drift=20, base_latency=1000, base_speed=3.0,
        base_mistakes=3, base_completion=0.90,
        drift_trend=0.70, latency_trend=0.60, speed_trend=0.55,
        mistake_trend=0.80, noise_scale=0.10,
    )

    # U004 — Mixed/Fluctuating (higher noise, no consistent trend)
    all_rows += generate_sessions(
        "U004", n_sessions=9,
        base_drift=28, base_latency=1400, base_speed=4.0,
        base_mistakes=6, base_completion=0.80,
        drift_trend=0.1, latency_trend=-0.05, speed_trend=0.08,
        noise_scale=0.25,
    )

    # U005 — Insufficient history (only 2 sessions — tests the fallback path)
    all_rows += generate_sessions(
        "U005", n_sessions=2,
        base_drift=30, base_latency=1500, base_speed=4.5,
        base_mistakes=7, base_completion=0.78,
    )

    df = pd.DataFrame(all_rows)

    # Shuffle slightly but keep within-user chronological order
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    path = "data/telemetry_sample.csv"
    df.to_csv(path, index=False)
    print(f"[OK] Synthetic dataset written → {path}")
    print(f"     Rows: {len(df)} | Users: {df['user_id'].nunique()} | Sessions: {df['session_id'].nunique()}")
    print()
    print("Per-user session counts:")
    print(df.groupby("user_id")["session_id"].count().to_string())


if __name__ == "__main__":
    main()
