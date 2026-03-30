"""
Pydantic schemas for input validation.
FastAPI automatically validates every incoming request against this.
If any field is missing or wrong type → 422 error returned automatically.
"""

from pydantic import BaseModel, Field
from typing import Literal


class ChurnPredictionInput(BaseModel):
    customerID: str = Field(default="UNKNOWN", description="Customer ID")
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=100)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(..., gt=0)
    TotalCharges: str = Field(default="0")


class ChurnPredictionOutput(BaseModel):
    customerID: str
    prediction: str
    churn_probability: float
    will_churn: bool
    message: str