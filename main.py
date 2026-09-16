import os
import mysql.connector
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from dotenv import load_dotenv

# Load database credentials from .env file
load_dotenv()

app = FastAPI(title="SIH26003 Cognitive Gaming Backend", version="1.0")

# --- Database Connection Helper ---
def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )

# --- Pydantic Models ---
class SessionStart(BaseModel):
    patient_id: int

class TelemetryLog(BaseModel):
    session_id: int
    interaction_type: str
    touch_drift_value: float
    hesitation_latency_ms: int

# --- API Endpoints ---
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