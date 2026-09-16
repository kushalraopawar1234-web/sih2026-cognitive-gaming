"""
api_client.py — Dashboard data access layer for SIH26003.

Function signatures are unchanged from Misba's original:
    get_patient_list()         → list of {patient_id, name}
    get_patient_data(patient_id) → full patient record

Data source:
    Primary : backend API (Kushal's FastAPI + Manjunath's ML endpoint)
    Fallback: sample_data.json (if backend is not running)

IMPORTANT DESIGN NOTES:
    - patient_id (dashboard) and user_id (ML) are kept as SEPARATE identifiers.
      No P001 == U001 mapping is assumed.
    - risk_flag in the dashboard UI is driven by the ML behavioral indicator.
      Mapping is neutral and observation-based only (not clinical):
          "Behavioral pattern within expected range"  → "low"
          "Moderate behavioral variation detected"    → "moderate"
          "Notable behavioral change detected"        → "elevated"
      The word "risk" here refers to the dashboard's technical field name only.
      It does NOT constitute a medical diagnosis or clinical risk prediction.
    - Radar chart axes: only genuinely matching ML fields are mapped.
      touch_drift      ← ML touch_drift_norm       (same concept)
      hesitation       ← ML response_latency_norm  (same concept, different label)
      decision_speed   ← ML decision_speed_norm    (same concept)
      task_variety     ← 0.0  (no equivalent in ML output — placeholder)
      sessions_compl.  ← 0.0  (no equivalent in ML output — placeholder)
    - The API key is never requested or stored here.

Project: AI-Based Cognitive Gaming and Memory Assistance Platform
SIH 2026 | Problem ID: SIH26003 | Team ID: 271
"""

import json
import os
import requests

# ── Backend URL ───────────────────────────────────────────────────────────────
# Set BACKEND_URL in environment to override, e.g.:
#   export BACKEND_URL=http://localhost:8000
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# ── Fallback local data (used when backend is not running) ────────────────────
_SAMPLE_DATA_FILE = os.path.join(os.path.dirname(__file__), "sample_data.json")

# ── Synthetic telemetry used for demo ML call ─────────────────────────────────
# These are representative sessions for 3 patients.
# In production these would be fetched from the backend database.
# user_id is kept separate from patient_id — no fabricated mapping.
_DEMO_SESSIONS = {
    "P001": [
        {"session_id": "P001_S001", "user_id": "P001", "timestamp": "2026-09-01 09:00:00",
         "touch_drift": 22.5, "response_latency": 1100.0, "decision_speed": 3.2,
         "mistakes": 4, "completion_rate": 0.88, "session_duration": 540.0},
        {"session_id": "P001_S002", "user_id": "P001", "timestamp": "2026-09-03 09:00:00",
         "touch_drift": 24.1, "response_latency": 1150.0, "decision_speed": 3.4,
         "mistakes": 3, "completion_rate": 0.90, "session_duration": 560.0},
        {"session_id": "P001_S003", "user_id": "P001", "timestamp": "2026-09-05 09:00:00",
         "touch_drift": 23.8, "response_latency": 1120.0, "decision_speed": 3.3,
         "mistakes": 5, "completion_rate": 0.87, "session_duration": 530.0},
        {"session_id": "P001_S004", "user_id": "P001", "timestamp": "2026-09-07 09:00:00",
         "touch_drift": 26.2, "response_latency": 1200.0, "decision_speed": 3.5,
         "mistakes": 4, "completion_rate": 0.85, "session_duration": 550.0},
    ],
    "P002": [
        {"session_id": "P002_S001", "user_id": "P002", "timestamp": "2026-09-01 10:00:00",
         "touch_drift": 18.0, "response_latency": 950.0, "decision_speed": 2.8,
         "mistakes": 2, "completion_rate": 0.93, "session_duration": 620.0},
        {"session_id": "P002_S002", "user_id": "P002", "timestamp": "2026-09-03 10:00:00",
         "touch_drift": 19.5, "response_latency": 980.0, "decision_speed": 2.9,
         "mistakes": 3, "completion_rate": 0.91, "session_duration": 600.0},
        {"session_id": "P002_S003", "user_id": "P002", "timestamp": "2026-09-05 10:00:00",
         "touch_drift": 21.0, "response_latency": 1000.0, "decision_speed": 3.0,
         "mistakes": 3, "completion_rate": 0.90, "session_duration": 590.0},
        {"session_id": "P002_S004", "user_id": "P002", "timestamp": "2026-09-07 10:00:00",
         "touch_drift": 20.5, "response_latency": 990.0, "decision_speed": 2.95,
         "mistakes": 2, "completion_rate": 0.92, "session_duration": 610.0},
    ],
    "P003": [
        {"session_id": "P003_S001", "user_id": "P003", "timestamp": "2026-09-01 11:00:00",
         "touch_drift": 35.0, "response_latency": 1800.0, "decision_speed": 5.5,
         "mistakes": 9, "completion_rate": 0.70, "session_duration": 480.0},
        {"session_id": "P003_S002", "user_id": "P003", "timestamp": "2026-09-03 11:00:00",
         "touch_drift": 38.5, "response_latency": 1950.0, "decision_speed": 5.8,
         "mistakes": 11, "completion_rate": 0.65, "session_duration": 460.0},
        {"session_id": "P003_S003", "user_id": "P003", "timestamp": "2026-09-05 11:00:00",
         "touch_drift": 42.0, "response_latency": 2100.0, "decision_speed": 6.2,
         "mistakes": 13, "completion_rate": 0.60, "session_duration": 440.0},
        {"session_id": "P003_S004", "user_id": "P003", "timestamp": "2026-09-07 11:00:00",
         "touch_drift": 45.5, "response_latency": 2300.0, "decision_speed": 6.5,
         "mistakes": 15, "completion_rate": 0.55, "session_duration": 420.0},
    ],
}

# ── Patient name registry ─────────────────────────────────────────────────────
# In production these names would come from the database.
_PATIENT_NAMES = {
    "P001": "Patient 1",
    "P002": "Patient 2",
    "P003": "Patient 3",
}


# ── Neutral indicator → dashboard field mapping ───────────────────────────────
def _indicator_to_observation_label(indicator: str) -> str:
    """
    Map the ML behavioral indicator string to a short observation label
    for the dashboard banner.

    This is a NEUTRAL, OBSERVATION-ONLY mapping.
    It does NOT constitute a medical diagnosis, clinical risk classification,
    or disease prediction. The label describes behavioral change relative
    to the user's own personal baseline only.
    """
    ind = indicator.lower()
    if "notable" in ind:
        return "elevated"    # Notable behavioral change vs personal baseline
    elif "moderate" in ind:
        return "moderate"    # Moderate behavioral variation vs personal baseline
    else:
        return "low"         # Within expected range of personal baseline


def _normalize_0_1(value: float, lo: float, hi: float) -> float:
    """Normalize a value to 0–1 given known range. Returns 0.5 if range is zero."""
    if hi == lo:
        return 0.5
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _ml_result_to_patient_record(patient_id: str, ml_results: list) -> dict:
    """
    Adapt the ML pipeline's output to the structure Misba's dashboard expects.

    Misba's app.py and charts.py expect:
        patient_id, name, engagement (list of {date, score}),
        telemetry (dict of 5 radar axes), risk_flag

    ML results provide:
        behavioral_score, indicator, trend, baseline_available,
        top_contributing_factors, explanation, metrics, timestamp

    Mapping rules (per user approval — only genuine matches):
        engagement.score  ← behavioral_score  (per-session behavioral score)
        telemetry.touch_drift       ← touch_drift_norm from metrics
        telemetry.hesitation_latency ← response_latency_norm  (same concept)
        telemetry.decision_speed    ← decision_speed_norm
        telemetry.task_variety      ← 0.0  (no ML equivalent — placeholder)
        telemetry.sessions_completed ← 0.0  (no ML equivalent — placeholder)
        risk_flag  ← mapped from indicator using neutral observation labels

    Normalization of radar axes: ML metrics are raw values, not 0–1 normalized
    at this layer. We use simple range normalization based on expected ranges
    from ml-analytics/src/config.py:
        touch_drift:       0 – 500 px
        response_latency:  0 – 30000 ms
        decision_speed:    0 – 60 s
    """
    engagement = []
    latest_result = None

    for r in sorted(ml_results, key=lambda x: x.get("timestamp", "")):
        date_str = str(r.get("timestamp", ""))[:10]  # YYYY-MM-DD
        score = r.get("behavioral_score", 0.5)
        engagement.append({"date": date_str, "score": round(score, 4)})
        latest_result = r

    # ── Radar telemetry from the latest session ────────────────────────────
    telemetry = {
        "touch_drift":        0.0,
        "hesitation_latency": 0.0,
        "decision_speed":     0.0,
        "task_variety":       0.0,   # No ML equivalent — placeholder
        "sessions_completed": 0.0,   # No ML equivalent — placeholder
    }
    if latest_result and latest_result.get("metrics"):
        m = latest_result["metrics"]
        # Normalize to 0–1 using ranges from config.py
        telemetry["touch_drift"]        = round(_normalize_0_1(m.get("touch_drift", 0),       0, 500), 4)
        telemetry["hesitation_latency"] = round(_normalize_0_1(m.get("response_latency", 0),  0, 30000), 4)
        telemetry["decision_speed"]     = round(_normalize_0_1(m.get("decision_speed", 0),    0, 60), 4)
        # task_variety and sessions_completed remain 0.0 — no genuine ML equivalent

    # ── Observation label from indicator ──────────────────────────────────
    obs_label = "low"
    if latest_result:
        obs_label = _indicator_to_observation_label(
            latest_result.get("indicator", "")
        )

    return {
        "patient_id": patient_id,
        "name": _PATIENT_NAMES.get(patient_id, patient_id),
        "engagement": engagement,
        "telemetry": telemetry,
        # risk_flag field is required by Misba's charts.py for the banner color.
        # Value is an observation label derived from the ML behavioral indicator.
        # It does NOT constitute a medical diagnosis or clinical risk prediction.
        "risk_flag": obs_label,
        # ── Extended ML fields (available for future dashboard expansion) ──
        "analytics": {
            "behavioral_score":         latest_result.get("behavioral_score") if latest_result else None,
            "indicator":                latest_result.get("indicator") if latest_result else None,
            "trend":                    latest_result.get("trend") if latest_result else None,
            "baseline_available":       latest_result.get("baseline_available") if latest_result else None,
            "top_contributing_factors": latest_result.get("top_contributing_factors") if latest_result else [],
            "explanation":              latest_result.get("explanation") if latest_result else None,
            "disclaimer":               latest_result.get("disclaimer") if latest_result else None,
        },
    }


def _fetch_from_backend(patient_id: str) -> dict:
    """
    Call the backend ML endpoint for a given patient_id.
    Returns the adapted patient record dict.
    Raises requests.RequestException if the backend is unavailable.
    """
    sessions = _DEMO_SESSIONS.get(patient_id, [])
    if not sessions:
        raise ValueError(f"No demo sessions defined for patient_id={patient_id}")

    response = requests.post(
        f"{BACKEND_URL}/api/v1/ml/analyze",
        json={"sessions": sessions},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    ml_results = data.get("results", [])
    return _ml_result_to_patient_record(patient_id, ml_results)


def _load_fallback_data() -> dict:
    """Load the local sample_data.json as a fallback."""
    with open(_SAMPLE_DATA_FILE, "r") as f:
        return json.load(f)


# ── Public API — unchanged function signatures ─────────────────────────────────

def get_patient_list() -> list:
    """Returns a list of {patient_id, name} for the dropdown selector."""
    return [
        {"patient_id": pid, "name": name}
        for pid, name in _PATIENT_NAMES.items()
    ]


def get_patient_data(patient_id: str) -> dict:
    """
    Returns the full record (engagement history, telemetry, risk_flag, analytics)
    for one patient.

    Attempts to fetch live ML results from the backend.
    Falls back to sample_data.json if the backend is unreachable.
    """
    try:
        return _fetch_from_backend(patient_id)
    except (requests.RequestException, requests.exceptions.ConnectionError) as e:
        # Backend not running — use local fallback so the dashboard still works standalone
        print(f"[api_client] Backend unavailable ({e}). Using sample_data.json fallback.")
        fallback = _load_fallback_data()
        for p in fallback["patients"]:
            if p["patient_id"] == patient_id:
                return p
        raise ValueError(f"No patient found with id {patient_id} in fallback data.")
    except ValueError as ve:
        raise ve