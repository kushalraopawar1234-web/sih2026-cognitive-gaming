"""
api_client.py
"""


import json
import os

DATA_FILE = os.path.join(os.path.dirname(__file__), "sample_data.json")


def _load_all_data():
    with open(DATA_FILE, "r") as f:
        return json.load(f)


def get_patient_list():
    """Returns a list of {patient_id, name} for the dropdown selector."""
    data = _load_all_data()
    return [{"patient_id": p["patient_id"], "name": p["name"]} for p in data["patients"]]


def get_patient_data(patient_id):
    """Returns the full record (engagement history, telemetry, risk_flag) for one patient."""
    data = _load_all_data()
    for p in data["patients"]:
        if p["patient_id"] == patient_id:
            return p
    raise ValueError(f"No patient found with id {patient_id}")