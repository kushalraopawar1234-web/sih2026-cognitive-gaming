"""
main.py — SIH26003 Cognitive Gaming Backend

Existing endpoints (Kushal — DO NOT MODIFY):
    GET  /                          Health check
    POST /api/v1/telemetry/log      Log telemetry micro-metric to MySQL

Integration endpoints (Manjunath — added for module integration):
    POST /api/v1/ml/analyze         Run ML behavioral analytics pipeline
    POST /api/v1/cultural/scenario  Generate cultural narrative scenario

Run with:
    uvicorn backend.main:app --reload

Project: AI-Based Cognitive Gaming and Memory Assistance Platform
SIH 2026 | Problem ID: SIH26003 | Team ID: 271

DISCLAIMER:
    This is a PROTOTYPE ONLY.
    It does NOT diagnose dementia or any medical condition.
    All behavioral indicators are for evaluator support only.
"""

import os
import sys
import json
import io
import mysql.connector
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Optional, Any, Dict

# ── Load .env ──────────────────────────────────────────────────────────────────
load_dotenv()

# ── Make ML src importable ─────────────────────────────────────────────────────
# ml-analytics/src/ is added to path so we can import the existing ML modules
# without modifying them.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ML_ROOT = os.path.join(_REPO_ROOT, "ml-analytics")
sys.path.insert(0, _ML_ROOT)

# ── Make Cultural AI src importable ───────────────────────────────────────────
_CULTURAL_SRC = os.path.join(_REPO_ROOT, "ai-cultural-narrative", "src")
sys.path.insert(0, _CULTURAL_SRC)

# ── Import existing ML functions (unchanged) ───────────────────────────────────
try:
    from src.preprocessing import validate, preprocess, engineer_features
    from src.baseline import compute_personal_baseline
    from src.scoring import compute_scores
    from src.analytics import detect_trends, build_results
    import pandas as pd
    _ML_AVAILABLE = True
except ImportError as _ml_err:
    _ML_AVAILABLE = False
    _ML_IMPORT_ERROR = str(_ml_err)

# ── Import existing Cultural AI functions (unchanged) ──────────────────────────
try:
    from scenario_generator import generate_scenario, CulturalDataError
    from llm_generator import LLMClient, MissingAPIKeyError, APIRequestError, InvalidLLMResponseError
    _CULTURAL_AVAILABLE = True
except ImportError as _ca_err:
    _CULTURAL_AVAILABLE = False
    _CA_IMPORT_ERROR = str(_ca_err)

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SIH26003 Cognitive Gaming Backend",
    version="1.1",
    description=(
        "PROTOTYPE ONLY — NOT a medical diagnostic system. "
        "Behavioral indicators are for evaluator support only."
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
# EXISTING CODE — Kushal's database connection (DO NOT MODIFY)
# ──────────────────────────────────────────────────────────────────────────────
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )

# ──────────────────────────────────────────────────────────────────────────────
# EXISTING Pydantic Models — Kushal (DO NOT MODIFY)
# ──────────────────────────────────────────────────────────────────────────────
class SessionStart(BaseModel):
    patient_id: int

class TelemetryLog(BaseModel):
    session_id: int
    interaction_type: str
    touch_drift_value: float
    hesitation_latency_ms: int

# ──────────────────────────────────────────────────────────────────────────────
# EXISTING API Endpoints — Kushal (DO NOT MODIFY)
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {"status": "Database-connected backend is running!"}

@app.post("/api/v1/telemetry/log")
def log_telemetry(log_data: TelemetryLog):
    db = None
    try:
        db = get_db_connection()
        cursor = db.cursor()

        # Insert the micro-metric data into MySQL
        sql = """INSERT INTO telemetry_logs
                 (session_id, interaction_type, touch_drift_value, hesitation_latency_ms)
                 VALUES (%s, %s, %s, %s)"""
        values = (log_data.session_id, log_data.interaction_type, log_data.touch_drift_value, log_data.hesitation_latency_ms)

        cursor.execute(sql, values)
        db.commit()

        return {"status": "success", "message": "Telemetry securely logged to MySQL."}
    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=str(err))
    finally:
        if db and db.is_connected():
            cursor.close()
            db.close()

# ──────────────────────────────────────────────────────────────────────────────
# NEW Pydantic Models — Integration (Manjunath)
# ──────────────────────────────────────────────────────────────────────────────

class TelemetrySession(BaseModel):
    """One gameplay session — matches the ML pipeline's required CSV columns."""
    session_id: str
    user_id: str
    timestamp: str                   # ISO datetime string e.g. "2026-09-01 10:00:00"
    touch_drift: float               # pixels of touch drift
    response_latency: float          # response time in milliseconds
    decision_speed: float            # seconds per decision
    mistakes: int                    # count of mistakes
    completion_rate: float           # fraction 0.0 – 1.0
    session_duration: float          # session length in seconds
    game_type: Optional[str] = None  # optional — kept but not used in scoring
    data_note: Optional[str] = None  # optional metadata

class MLAnalyzeRequest(BaseModel):
    """
    Request body for POST /api/v1/ml/analyze.
    Submit one or more sessions for the same or multiple users.
    At least 3 prior sessions per user are needed for a personal baseline.
    With fewer sessions, baseline_available = false and a fallback score is used.
    No sessions are fabricated.
    """
    sessions: List[TelemetrySession]

class PatientProfile(BaseModel):
    """Patient cultural profile for scenario generation."""
    age: int
    region: str           # e.g. "Assam" — must exist in ner_culture.json
    language: str         # e.g. "English" — must be in region's supported languages
    festival: str         # e.g. "Bihu" — must be in region's verified festivals
    favourite_food: str
    childhood_activity: str
    preferred_topic: str

class CulturalScenarioRequest(BaseModel):
    """Request body for POST /api/v1/cultural/scenario."""
    patient_profile: PatientProfile
    use_demo_llm: bool = True  # True = use offline DemoLLM (no API key required)
                               # False = use real LLM (requires LLM_API_KEY env var)

# ──────────────────────────────────────────────────────────────────────────────
# Demo LLM — offline stub for testing without a real API key
# Mirrors the exact interface of LLMClient from llm_generator.py
# Returns a structurally valid scenario using the profile's own data
# ──────────────────────────────────────────────────────────────────────────────
class _DemoLLM:
    """
    Offline stub that mirrors LLMClient.generate_json().
    Used when use_demo_llm=True or when LLM_API_KEY is not set.
    Returns a structurally valid response using the profile fields.
    No real LLM call is made. No API key is needed.
    """
    def __init__(self, profile: dict):
        self._profile = profile

    def generate_json(self, prompt: str) -> dict:
        festival = self._profile.get("festival", "a local festival")
        food     = self._profile.get("favourite_food", "home cooking")
        activity = self._profile.get("childhood_activity", "community activities")
        topic    = self._profile.get("preferred_topic", "family memories")
        region   = self._profile.get("region", "the region")
        return {
            "title": f"{festival} Memories",
            "scenario": (
                f"{festival} is a time that can bring thoughts of {topic}. "
                f"You may skip this question if you prefer."
            ),
            "question": (
                f"Do you have any memories of {festival} that you would like to share, "
                f"if you feel comfortable?"
            ),
            "follow_up_topics": [activity, food, "family"],
        }

# ──────────────────────────────────────────────────────────────────────────────
# NEW Endpoint — ML Behavioral Analytics
# POST /api/v1/ml/analyze
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/api/v1/ml/analyze")
def ml_analyze(request: MLAnalyzeRequest):
    """
    Run the ML behavioral analytics pipeline on provided session telemetry.

    Pipeline (unchanged from ml-analytics/src/):
        validate → preprocess → engineer_features →
        compute_personal_baseline → compute_scores → detect_trends → build_results

    Returns per-session behavioral analytics including:
        behavioral_score, indicator, trend, baseline_available,
        top_contributing_factors, explanation

    DISCLAIMER: Prototype only. Not a medical diagnostic system.
    Does not diagnose dementia or any medical condition.
    """
    if not _ML_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=f"ML module could not be loaded. Check ml-analytics/src/ dependencies. Error: {_ML_IMPORT_ERROR}"
        )

    if not request.sessions:
        raise HTTPException(status_code=422, detail="sessions list must not be empty.")

    # ── Convert request to the DataFrame format the ML pipeline expects ──
    try:
        rows = [s.model_dump() for s in request.sessions]
        # Remove optional fields that the ML pipeline doesn't need as required columns
        df_raw = pd.DataFrame(rows)

        # Rename columns if necessary — the ML pipeline uses exactly these names:
        # session_id, user_id, timestamp, touch_drift, response_latency,
        # decision_speed, mistakes, completion_rate, session_duration
        # Our Pydantic model already uses these exact names, so no renaming needed.

    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse sessions into DataFrame: {e}")

    # ── Run existing ML pipeline (all functions unchanged) ──
    try:
        # Suppress ML pipeline's print statements to keep API response clean
        # (the pipeline prints step-by-step logs designed for CLI use)
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        df, issues = validate(df_raw)
        df = preprocess(df)
        df = engineer_features(df)
        df = compute_personal_baseline(df)
        df = compute_scores(df)
        df = detect_trends(df)
        results = build_results(df)

        sys.stdout = old_stdout

    except ValueError as ve:
        sys.stdout = old_stdout
        raise HTTPException(status_code=422, detail=f"ML validation error: {ve}")
    except Exception as e:
        sys.stdout = old_stdout
        raise HTTPException(status_code=500, detail=f"ML pipeline error: {e}")

    # ── Return only the fields relevant to the dashboard ──
    # Full result dict is returned — dashboard/api_client.py selects what it needs.
    # No risk classification. No medical labels. Neutral terminology only.
    return {
        "status": "ok",
        "sessions_analyzed": len(results),
        "disclaimer": (
            "PROTOTYPE ONLY — NOT a medical diagnostic system. "
            "Behavioral indicators are for evaluator support only. "
            "Does not diagnose dementia or any medical condition."
        ),
        "validation_issues": issues,
        "results": results,
    }


# ──────────────────────────────────────────────────────────────────────────────
# NEW Endpoint — Cultural Narrative Scenario
# POST /api/v1/cultural/scenario
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/api/v1/cultural/scenario")
def cultural_scenario(request: CulturalScenarioRequest):
    """
    Generate a culturally grounded reminiscence scenario for a patient.

    Uses Manoj's ai-cultural-narrative/src/scenario_generator.py (unchanged).

    If use_demo_llm=True (default): uses the offline DemoLLM stub.
        No API key required. Safe for testing and demos.

    If use_demo_llm=False: uses the real LLM via environment variable LLM_API_KEY.
        Requires LLM_API_KEY, LLM_API_URL, LLM_MODEL to be set.
        The API key is NEVER returned in any response.

    Cultural validation (region, festival, language) is always enforced
    regardless of which LLM is used.
    """
    if not _CULTURAL_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=f"Cultural AI module could not be loaded. Check ai-cultural-narrative/src/. Error: {_CA_IMPORT_ERROR}"
        )

    profile_dict = request.patient_profile.model_dump()

    # ── Choose LLM client ──────────────────────────────────────────────────────
    if request.use_demo_llm:
        llm_client = _DemoLLM(profile=profile_dict)
    else:
        # Real LLM — key is read from environment variable, never from request body
        llm_key = os.getenv("LLM_API_KEY")
        if not llm_key:
            raise HTTPException(
                status_code=503,
                detail=(
                    "LLM_API_KEY is not configured. Set it as an environment variable. "
                    "To test without a real key, send use_demo_llm=true."
                )
            )
        llm_client = LLMClient()  # reads LLM_API_KEY, LLM_API_URL, LLM_MODEL from env

    # ── Call the existing scenario_generator (unchanged) ──────────────────────
    try:
        result = generate_scenario(profile_dict, llm_client=llm_client)
    except CulturalDataError as cde:
        # Region or festival not in the verified dataset
        raise HTTPException(status_code=422, detail=f"Cultural data error: {cde}")
    except MissingAPIKeyError:
        raise HTTPException(
            status_code=503,
            detail="LLM_API_KEY is not configured. Set it as an environment variable."
        )
    except APIRequestError as are:
        raise HTTPException(status_code=502, detail=f"LLM provider request failed: {are}")
    except InvalidLLMResponseError as ilre:
        raise HTTPException(status_code=502, detail=f"LLM returned an invalid response: {ilre}")
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=f"Profile validation error: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cultural scenario generation error: {e}")

    # ── Return the scenario — LLM key is never included ───────────────────────
    return {
        "status": "ok",
        "title":            result["title"],
        "scenario":         result["scenario"],
        "question":         result["question"],
        "follow_up_topics": result["follow_up_topics"],
        "note": (
            "Culturally grounded reminiscence prompt. "
            "Not a medical tool. Patient may skip at any time."
        ),
    }