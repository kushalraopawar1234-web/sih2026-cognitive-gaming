# ML & Predictive Analytics — Work Log

> **Last Updated:** 2026-09-16
> **Maintained by:** ML & Predictive Analytics Lead
> **Status: STANDALONE PIPELINE COMPLETE — All 16 tests passing.**

---

## Project

**AI-Based Cognitive Gaming and Memory Assistance Platform for Elderly Dementia Patients in North Eastern Region (NER)**

---

## SIH Details

| Field | Value |
|---|---|
| Event | Smart India Hackathon 2026 |
| Problem Statement ID | SIH26003 |
| Theme | Healthcare |
| Category | Software |
| Team ID | 271 |
| ML Lead Role | ML & Predictive Analytics Lead |

---

## My Responsibility

- Telemetry analysis
- Touch drift analysis
- Response latency analysis
- Decision speed analysis
- Mistake / error analysis
- Engagement analysis
- Feature engineering
- Personal behavioral baseline
- Behavioral scoring
- Behavioral trend detection
- Explainable behavioral indicator
- JSON + CSV output for future dashboard integration

---

## Current Architecture

```
Game
  ↓
Telemetry
  ↓
Backend / Database  (Kushal's module — integration pending)
  ↓
ML Analytics        ← THIS MODULE (complete)
  ↓
Behavioral Score + Indicator
  ↓
Dashboard           (Misba's module — integration pending)
```

---

## Completed Work

- [x] Project structure
- [x] Sample telemetry dataset
- [x] Data validation
- [x] Preprocessing
- [x] Feature engineering
- [x] Personal baseline
- [x] Behavioral scoring
- [x] Trend detection
- [x] Visualization (5 charts)
- [x] JSON output
- [x] CSV output
- [x] API-ready interface (documented)
- [x] Testing (16/16 tests pass)
- [x] Documentation (README.md)

---

## Files Created

| File | Purpose | Status |
|---|---|---|
| `data/telemetry_sample.csv` | Synthetic telemetry — 5 users, 44 sessions | Done |
| `src/config.py` | All weights, thresholds, paths | Done |
| `src/generate_data.py` | Synthetic data generator script | Done |
| `src/preprocessing.py` | Load + Validate + Preprocess + Feature Engineering | Done |
| `src/baseline.py` | Personal behavioral baseline per user | Done |
| `src/scoring.py` | Behavioral scoring engine (weighted delta + sigmoid) | Done |
| `src/analytics.py` | Trend detection + JSON/CSV output + 5 charts | Done |
| `main.py` | Main pipeline orchestrator (one command) | Done |
| `requirements.txt` | Python dependencies | Done |
| `README.md` | Full project documentation | Done |
| `tests/test_pipeline.py` | 16 unit tests — all passing | Done |
| `outputs/results.json` | Full JSON output per session | Auto-generated |
| `outputs/scores.csv` | Scores CSV for quick review | Auto-generated |
| `outputs/charts/*.png` | 5 behavioral visualization charts | Auto-generated |
| `WORK_LOG.md` | AI handoff document | Maintained |

---

## Important Technical Decisions

- **Personal baseline over population thresholds:** Each user's own historical sessions are used as their baseline. More appropriate for elderly patients with high individual variability.
- **Sigmoid-transformed weighted delta scoring:** Score = sigmoid(sum of weight_i x delta_pct_i x 0.05). Maps any weighted change to [0,1]. Fully explainable.
- **Fallback scoring for insufficient history:** When a user has fewer than 3 previous sessions, a direct weighted-normalized score is used and clearly labelled as [No baseline].
- **Simple interpretable methods only:** Pandas, NumPy, Matplotlib — no TensorFlow, PyTorch, or LSTM.
- **Synthetic data for initial development:** 5 users, 44 sessions, 4 behavioral patterns.
- **No medical diagnosis claims:** System produces behavioral indicators only. Disclaimer embedded in every JSON result.
- **JSON as primary output format:** Structured for direct dashboard consumption by Misba.
- **All weights in config.py:** Judges can see and adjust scoring weights in one file.
- **Neutral language throughout:** "Notable behavioral change detected" — never "dementia progression."
- **python3 command on this Mac:** The command python does not work; always use python3.

---

## Current Input Format

**Telemetry CSV — currently expected columns:**

| Field | Type | Description |
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

Optional columns (kept but not used in scoring): `game_type`, `data_note`

---

## Current Output Format

**outputs/results.json** — one entry per session:

```json
{
  "session_id": "U003_S011",
  "user_id": "U003",
  "timestamp": "2026-07-21 00:00:00",
  "game_type": "memory_match",
  "metrics": {
    "touch_drift": 31.37,
    "response_latency": 1562.11,
    "decision_speed": 4.42,
    "mistakes": 5,
    "mistake_rate": 0.039,
    "completion_rate": 0.89,
    "session_duration": 566.64,
    "engagement_score": 0.53
  },
  "baseline_available": true,
  "baseline": { "touch_drift": 25.88, "response_latency": 1336.3 },
  "baseline_deltas_pct": { "touch_drift": 21.22, "response_latency": 16.9 },
  "behavioral_score": 0.7513,
  "score_method": "baseline_delta",
  "indicator": "Notable behavioral change detected",
  "trend": "Changing",
  "top_contributing_factors": [
    "mistake rate increased by 49.9% vs personal baseline",
    "touch drift increased by 21.2% vs personal baseline"
  ],
  "explanation": "Compared to personal baseline (10 previous sessions): ...",
  "disclaimer": "PROTOTYPE ONLY — NOT a medical diagnostic system."
}
```

Indicator values: Behavioral pattern within expected range | Moderate behavioral variation detected | Notable behavioral change detected

Trend values: Improving | Stable | Changing | Insufficient Data

---

## How to Run

```bash
# 1. Go to project root
cd /Users/manjunathn/Desktop/SIH

# 2. Install dependencies (one time only)
pip3 install -r requirements.txt

# 3. Run the full pipeline
python3 main.py

# 4. Analyze a single user
python3 main.py --user U001

# 5. Use a custom data file
python3 main.py --data path/to/telemetry.csv

# 6. Run all tests
python3 tests/test_pipeline.py

# 7. Regenerate synthetic dataset
python3 src/generate_data.py
```

---

## Current Status

**2026-09-16 — Pipeline COMPLETE & VERIFIED:**
Full standalone ML analytics pipeline built, tested, and passed final verification. Runs with `python3 main.py`. 
- **Tests Executed:** 16 unit tests run using pytest (`test_pipeline.py`).
- **Test Result:** 16/16 tests PASSED. All warnings fixed.
- **Demo Result:** Full pipeline executes cleanly.
- **Output Files Verified:** `outputs/results.json`, `outputs/scores.csv`, and 5 charts in `outputs/charts/` are generated correctly.
- **Personal Baseline Verified:** Correctly calculated based on prior history. Insufficient history gracefully defaults to raw normalized score without fabricating history.
- **Remaining Issues:** Integration with actual backend needed; current data is synthetic.
- **Next Step:** Coordinate with Kushal for backend DB/API schema to create `src/data_connector.py`.

---

## Pending Work

- [ ] Integrate with Kushal's backend database (replace CSV with DB/API call)
- [ ] Integrate JSON output with Misba's dashboard (provide API endpoint or file path)
- [ ] Confirm final telemetry schema with backend team before using real data
- [ ] Add a simple Flask/FastAPI wrapper if real-time API endpoint is needed

---

## Known Issues

| Issue | Severity | Status |
|---|---|---|
| Synthetic data only — no real patient data | Low (expected) | Open |
| Input schema not yet confirmed with backend (Kushal) | Medium | Open |
| Normalization is user-relative; new users need 3+ sessions for baseline | Low | By design |
| `python` command not available on this Mac — use `python3` | Low | Known |

---

## Future Backend Integration

**Required from Kushal:**

| Item | Description |
|---|---|
| Database type | PostgreSQL / MongoDB / Firebase? (TBD) |
| API endpoint or DB connection | To pull session telemetry per patient |
| Telemetry schema | Exact field names and types |
| Patient ID format | How user IDs are structured |
| Session frequency | How often sessions are recorded |
| Historical data window | How many past sessions to retrieve |
| Authentication method | API key / JWT / none? |
| Real-time vs batch | Per session or in batches? |

**When ready:** Update `src/config.py` with connection details and add a new `src/data_connector.py` module that fetches data and returns a DataFrame matching the current input schema. `main.py` will not need to change.

---

## Dashboard Integration

**ML module provides (per session):**

| Output Field | Type | For Dashboard |
|---|---|---|
| `behavioral_score` | float 0–1 | Primary metric gauge/bar |
| `indicator` | string | Status badge |
| `trend` | string | Trend arrow/icon |
| `top_contributing_factors` | list | Bullet list |
| `explanation` | string | Detail text block |
| `metrics` | object | Raw values table |
| `baseline_deltas_pct` | object | Comparison bars |
| `baseline_available` | bool | Show/hide baseline UI |

**Delivery:** `outputs/results.json` file, or REST API endpoint (to be added with Flask/FastAPI if needed).

**Misba should confirm:**
- Does the dashboard consume JSON files, or call an API?
- What refresh rate / polling interval is needed?
- Which fields are most critical for the patient view vs caregiver view?

---

# NEXT AI — READ THIS FIRST

## 1. What Has Already Been Completed
- Complete standalone ML analytics pipeline — all stages built and working.
- 16/16 unit tests passing.
- Full outputs: results.json, scores.csv, 5 charts.
- README.md and WORK_LOG.md fully up to date.

## 2. What Is Currently Working
Run `python3 main.py` from `/Users/manjunathn/Desktop/SIH/`. It loads and validates telemetry, preprocesses and engineers features, computes personal behavioral baselines, scores each session, detects trends, and saves all outputs.

## 3. What Should Be Done Next
- Confirm final telemetry column names with Kushal (backend lead).
- When Kushal's backend is ready: add `src/data_connector.py` to replace CSV loading with API/DB call.
- When Misba's dashboard is ready: provide `outputs/results.json` path or set up a Flask API.
- Optional: add Streamlit demo UI for SIH presentation.

## 4. Important Technical Decisions
- Personal baseline (not population) — each user is their own reference.
- Score = sigmoid(weighted % delta from baseline). Transparent and explainable.
- Fallback raw-normalized score when history is under 3 sessions.
- No deep learning — only Pandas, NumPy, Matplotlib.
- No diagnosis — only behavioral indicators.
- All weights configurable in src/config.py.
- python3 is the command on this Mac (not python).

## 5. Files That Must NOT Be Unnecessarily Rewritten
- WORK_LOG.md — update only, never replace.
- src/config.py — only change values, not structure.
- outputs/results.json / outputs/scores.csv — auto-generated, do not hand-edit.

## 6. Known Problems
- Real patient data not yet available — synthetic data only.
- Backend telemetry schema not yet finalized with Kushal.
- python command not found on this Mac — always use python3.

## 7. Exact Command to Run the Project
```bash
cd /Users/manjunathn/Desktop/SIH
python3 main.py
```

## 8. Backend Integration Status
NOT started. Pipeline is standalone. When Kushal provides the API/DB connection, create `src/data_connector.py` to fetch telemetry as a DataFrame. All other code stays the same.
