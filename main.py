from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

app = FastAPI(title="SIH26003 Cognitive Gaming Backend", version="1.0")

# --- Pydantic Models for Data Validation ---
class SessionStart(BaseModel):
    patient_id: int

class TelemetryLog(BaseModel):
    session_id: int
    interaction_type: str
    touch_drift_value: float
    hesitation_latency_ms: int

# --- Mock API Endpoints ---

@app.get("/")
def read_root():
    return {"status": "Backend is running", "message": "Welcome to the SIH26003 API"}

@app.post("/api/v1/session/start")
def start_session(session_data: SessionStart):
    # MOCK RESPONSE: Returning a dummy session ID and procedural theme
    return {
        "status": "success",
        "session_id": 101, 
        "patient_id": session_data.patient_id,
        "scenario_theme": "Hornbill Festival Environment",
        "start_time": datetime.now().isoformat()
    }

@app.post("/api/v1/telemetry/log")
def log_telemetry(log_data: TelemetryLog):
    # MOCK RESPONSE: Simulating a successful database insertion
    return {
        "status": "logged",
        "recorded_data": log_data,
        "message": "Telemetry micro-metrics captured successfully."
    }