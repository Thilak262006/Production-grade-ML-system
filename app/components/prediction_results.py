"""
Prediction Results Component — displays prediction outcome and charts.
"""

import streamlit as st
import plotly.graph_objects as go


def render_prediction_results(result: dict):
    """Render prediction result with visual indicators."""

    st.subheader("Prediction Result")

    # ── Main result card ──────────────────────────────────────────────────────
    will_churn = result["will_churn"]
    probability = result["churn_probability"]
    prediction = result["prediction"]
    message = result["message"]

    if will_churn:
        st.error(f"⚠️ **CHURN RISK DETECTED**")
        color = "red"
    else:
        st.success(f"✅ **LOW CHURN RISK**")
        color = "green"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prediction", prediction)
    with col2:
        st.metric("Churn Probability", f"{probability * 100:.1f}%")
    with col3:
        st.metric("Risk Level",
                  "High" if probability > 0.7
                  else "Medium" if probability > 0.4
                  else "Low")

    st.info(f"💡 {message}")

    # ── Gauge chart ───────────────────────────────────────────────────────────
    st.subheader("Churn Probability Gauge")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={"text": "Churn Probability (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40],   "color": "lightgreen"},
                {"range": [40, 70],  "color": "yellow"},
                {"range": [70, 100], "color": "lightsalmon"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 70
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)