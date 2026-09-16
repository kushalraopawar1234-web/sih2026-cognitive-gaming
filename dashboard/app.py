"""
app.py — Clinician Dashboard for SIH26003

Run with:  streamlit run dashboard/app.py   (from the repo root)
"""

import streamlit as st
from api_client import get_patient_list, get_patient_data
from charts import build_retention_curve, build_radar_chart, risk_flag_display

st.set_page_config(page_title="Clinician Dashboard - SIH26003", layout="wide")

st.title("🧠 Clinician Dashboard")
st.caption("Cognitive engagement & risk overview for reviewing physicians")

# ---------- Patient selector ----------
patients = get_patient_list()
patient_names = {p["patient_id"]: p["name"] for p in patients}

selected_id = st.selectbox(
    "Select patient",
    options=list(patient_names.keys()),
    format_func=lambda pid: patient_names[pid],
)

patient_data = get_patient_data(selected_id)

# ---------- Risk flag banner ----------
color, label = risk_flag_display(patient_data["risk_flag"])
st.markdown(
    f"""
    <div style="background-color:{color}; padding:10px 16px; border-radius:6px;
                color:white; font-weight:bold; display:inline-block; margin-bottom:20px;">
        ⚠ {label}
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Charts side by side ----------
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(build_retention_curve(patient_data), use_container_width=True)

with col2:
    st.plotly_chart(build_radar_chart(patient_data), use_container_width=True)

st.divider()
st.caption(
    "Data source: placeholder sample_data.json. Will switch to live backend "
    "once Kushal's API is ready — see api_client.py for the swap point."
)