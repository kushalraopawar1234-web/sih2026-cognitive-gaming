"""
charts.py
"""


import plotly.graph_objects as go


def build_retention_curve(patient_data):
    dates = [e["date"] for e in patient_data["engagement"]]
    scores = [e["score"] for e in patient_data["engagement"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=scores,
        mode="lines+markers",
        line=dict(color="#1a3c6e", width=3),
        marker=dict(size=7),
        name="Engagement score",
    ))
    fig.update_layout(
        title=f"Engagement Retention — {patient_data['name']}",
        xaxis_title="Date",
        yaxis_title="Engagement score (0-1)",
        yaxis_range=[0, 1],
        height=380,
    )
    return fig


def build_radar_chart(patient_data):
    telemetry = patient_data["telemetry"]
    # Order matters for how the radar looks - keep axes consistent across patients
    axes = ["touch_drift", "hesitation_latency", "decision_speed", "task_variety", "sessions_completed"]
    labels = ["Touch Drift", "Hesitation Latency", "Decision Speed", "Task Variety", "Sessions Completed"]

    values = [telemetry[a] for a in axes]
    # Close the loop for the radar shape
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        line=dict(color="#1a3c6e"),
        name="Data capture depth",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f"Data Capture Depth — {patient_data['name']}",
        height=420,
        showlegend=False,
    )
    return fig


def risk_flag_display(risk_flag):
    """Returns (color, label) for rendering a risk badge in app.py."""
    mapping = {
        "low": ("#2e7d32", "LOW RISK"),
        "medium": ("#f9a825", "MEDIUM RISK"),
        "high": ("#c62828", "HIGH RISK"),
    }
    return mapping.get(risk_flag.lower(), ("#757575", risk_flag.upper()))