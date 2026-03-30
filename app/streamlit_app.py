"""
Streamlit Web UI — Customer Churn Prediction App.
Sends requests to the FastAPI backend and displays results.
"""

import os
import requests
import streamlit as st
from dotenv import load_dotenv
from components.input_form import render_input_form
from components.prediction_results import render_prediction_results

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Customer Churn Prediction System")
st.markdown("Fill in the customer details below and click **Predict** to see the churn risk.")
st.divider()

# ── API Settings ──────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("API_SECRET_KEY", "my-secret-churn-api-key-2024")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ API Settings")
    api_url = st.text_input("API URL", value=API_URL)
    api_key = st.text_input("API Key", value=API_KEY, type="password")
    st.divider()
    st.header("📈 Model Info")

    # Check API health
    try:
        health = requests.get(f"{api_url}/health", timeout=3)
        if health.status_code == 200:
            data = health.json()
            st.success("API Status: Healthy ✅")
            st.info(f"Model: {data['model']}")
            st.info(f"Version: {data['version']}")
        else:
            st.error("API Status: Unhealthy ❌")
    except Exception:
        st.error("API not reachable ❌")
        st.warning("Make sure API is running:\nuvicorn api.api_server:app --reload")

# ── Input Form ────────────────────────────────────────────────────────────────
input_data = render_input_form()

st.divider()

# ── Predict Button ────────────────────────────────────────────────────────────
if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
    with st.spinner("Analyzing customer data..."):
        try:
            response = requests.post(
                f"{api_url}/predict",
                json=input_data,
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": api_key
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                st.divider()
                render_prediction_results(result)
            else:
                st.error(f"API Error {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure it is running!")
        except Exception as e:
            st.error(f"Error: {str(e)}")