"""
Input form component — renders all customer input fields.
"""

import streamlit as st


def render_input_form() -> dict:
    """Render the customer input form and return values as dict."""

    st.subheader("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        customerID = st.text_input("Customer ID", value="TEST-001")
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        MonthlyCharges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        TotalCharges = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0)

    with col2:
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox(
            "Multiple Lines", ["Yes", "No", "No phone service"]
        )
        InternetService = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        OnlineSecurity = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        OnlineBackup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        DeviceProtection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        TechSupport = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )

    with col3:
        StreamingTV = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        StreamingMovies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )
        Contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        PaymentMethod = st.selectbox(
            "Payment Method", [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    return {
        "customerID"      : customerID,
        "gender"          : gender,
        "SeniorCitizen"   : SeniorCitizen,
        "Partner"         : Partner,
        "Dependents"      : Dependents,
        "tenure"          : tenure,
        "PhoneService"    : PhoneService,
        "MultipleLines"   : MultipleLines,
        "InternetService" : InternetService,
        "OnlineSecurity"  : OnlineSecurity,
        "OnlineBackup"    : OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport"     : TechSupport,
        "StreamingTV"     : StreamingTV,
        "StreamingMovies" : StreamingMovies,
        "Contract"        : Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod"   : PaymentMethod,
        "MonthlyCharges"  : MonthlyCharges,
        "TotalCharges"    : str(TotalCharges),
    }