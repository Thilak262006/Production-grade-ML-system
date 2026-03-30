import pytest
import pandas as pd
from src.models.model_prediction import ModelPredictor


@pytest.fixture
def sample_customer():
    return pd.DataFrame([{
        "customerID"      : "TEST-001",
        "gender"          : "Male",
        "SeniorCitizen"   : 0,
        "Partner"         : "Yes",
        "Dependents"      : "No",
        "tenure"          : 2,
        "PhoneService"    : "Yes",
        "MultipleLines"   : "No",
        "InternetService" : "Fiber optic",
        "OnlineSecurity"  : "No",
        "OnlineBackup"    : "No",
        "DeviceProtection": "No",
        "TechSupport"     : "No",
        "StreamingTV"     : "No",
        "StreamingMovies" : "No",
        "Contract"        : "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod"   : "Electronic check",
        "MonthlyCharges"  : 70.35,
        "TotalCharges"    : "140.70",
        "Churn"           : "No"
    }])


def test_predictor_loads_artifacts():
    predictor = ModelPredictor()
    predictor.load_artifacts()
    assert predictor.model is not None
    assert predictor.transformer is not None
    assert predictor.label_encoder is not None


def test_predictor_returns_correct_keys(sample_customer):
    predictor = ModelPredictor()
    predictor.load_artifacts()
    result = predictor.predict(sample_customer)
    assert "prediction" in result
    assert "churn_probability" in result
    assert "will_churn" in result


def test_predictor_probability_range(sample_customer):
    predictor = ModelPredictor()
    predictor.load_artifacts()
    result = predictor.predict(sample_customer)
    assert 0.0 <= result["churn_probability"] <= 1.0


def test_predictor_prediction_values(sample_customer):
    predictor = ModelPredictor()
    predictor.load_artifacts()
    result = predictor.predict(sample_customer)
    assert result["prediction"] in ["Yes", "No"]