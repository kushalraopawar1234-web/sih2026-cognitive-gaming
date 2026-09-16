# ML & Predictive Analytics Module
## AI-Based Cognitive Gaming and Memory Assistance Platform
### SIH 2026 | Problem ID: SIH26003 | Team ID: 271

> **PROTOTYPE ONLY — NOT a medical diagnostic system.**
> This module analyzes gameplay behavior and highlights behavioral changes for evaluator support.
> It does not diagnose dementia or any medical condition.

---

## What This Module Does

This is the **ML & Predictive Analytics** component of our SIH 2026 project.

It takes gameplay telemetry data from cognitive games and:
1. **Validates** the incoming data for quality
2. **Preprocesses** and engineers behavioral features
3. **Computes a personal behavioral baseline** for each user (from their own history)
4. **Scores** each session against that personal baseline
5. **Detects behavioral trends** (Improving / Stable / Changing)
6. **Generates explainable outputs** — JSON + CSV + charts
7. **Flags notable changes** for further evaluator review

---

## Key Differentiator — Adaptive Personal Baseline

Instead of comparing users against a fixed population average,
**each user is compared only to their own previous sessions.**

```
Previous Sessions → Personal Baseline
                            ↓
            Current Session (compared to baseline)
                            ↓
              Behavioral Change Score + Explanation
```

This is more appropriate for elderly patients, where individual
baselines vary greatly.

---

## Quick Start

```bash
# 1. Install dependencies
pip3 install -r requirements.txt

# 2. Run the full pipeline
python3 main.py

# 3. Analyze a single user
python3 main.py --user U001

# 4. Use a custom data file
python3 main.py --data path/to/your/telemetry.csv

# 5. Run tests
python3 tests/test_pipeline.py
```

---

## Project Structure

```
SIH/
├── data/
│   └── telemetry_sample.csv     ← Synthetic demo data (NOT clinical)
│
├── src/
│   ├── config.py                ← All weights, thresholds, paths
│   ├── generate_data.py         ← Synthetic data generator
│   ├── preprocessing.py         ← Validation + Preprocessing + Features
│   ├── baseline.py              ← Personal behavioral baseline
│   ├── scoring.py               ← Behavioral scoring engine
│   └── analytics.py             ← Trend detection + output + charts
│
├── outputs/
│   ├── results.json             ← Full JSON output (for dashboard)
│   ├── scores.csv               ← Scores table (for review)
│   └── charts/                  ← 5 visualization charts
│
├── tests/
│   └── test_pipeline.py         ← 16 unit tests (all passing)
│
├── main.py                      ← Main pipeline (one command)
├── requirements.txt
├── README.md
└── WORK_LOG.md                  ← AI handoff document
```

---

## Input Format

The pipeline expects a CSV with these columns:

| Column | Type | Description |
|---|---|---|
| `session_id` | string | Unique session identifier |
| `user_id` | string | Unique patient identifier |
| `timestamp` | datetime | When the session occurred |
| `touch_drift` | float | Pixels of average touch drift |
| `response_latency` | float | Response time in milliseconds |
| `decision_speed` | float | Seconds per decision |
| `mistakes` | int | Count of mistakes in session |
| `completion_rate` | float | Fraction of game completed (0–1) |
| `session_duration` | float | Session length in seconds |

---

## Output Format

`outputs/results.json` — one entry per session:

```json
{
  "session_id": "U003_S011",
  "user_id": "U003",
  "behavioral_score": 0.7513,
  "indicator": "Notable behavioral change detected",
  "trend": "Changing",
  "baseline_available": true,
  "top_contributing_factors": [
    "mistake rate increased by 49.9% vs personal baseline",
    "touch drift increased by 21.2% vs personal baseline"
  ],
  "explanation": "...",
  "disclaimer": "PROTOTYPE ONLY..."
}
```

---

## Behavioral Score

| Score Range | Indicator |
|---|---|
| 0.00 – 0.44 | Behavioral pattern within expected range |
| 0.45 – 0.69 | Moderate behavioral variation detected |
| 0.70 – 1.00 | Notable behavioral change detected |

---

## Trend Labels

| Label | Meaning |
|---|---|
| Improving | Recent sessions score consistently lower (better) than baseline |
| Stable | Recent and baseline scores are similar |
| Changing | Recent sessions score consistently higher (more changed) than baseline |
| Insufficient Data | Not enough sessions to compute trend |

---

## Backend Integration (Future — Kushal)

When Kushal's backend is ready, replace `data/telemetry_sample.csv` with
data fetched from the backend API. Update `src/config.py` with the
API endpoint / DB connection details.

The pipeline interface will remain unchanged — it only needs a DataFrame
with the columns listed in "Input Format" above.

---

## Dashboard Integration (Future — Misba)

The `outputs/results.json` file is structured for direct dashboard consumption.
Key fields for the UI:
- `behavioral_score` — primary metric
- `indicator` — display as a badge/status
- `trend` — display as trend arrow/icon
- `top_contributing_factors` — display as a bullet list
- `explanation` — display as a detail text block

---

## Disclaimer

This module is a **prototype analytics system** built for SIH 2026 demonstration.

- It analyzes gameplay behavior patterns only
- It does NOT diagnose dementia or any medical condition
- All current data is synthetic (not real patient data)
- Thresholds and weights are configurable and not clinically validated
- Clinical decisions must always involve qualified medical professionals
