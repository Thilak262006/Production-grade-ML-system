"""
Prediction Route — POST /predict
Takes customer data → returns churn prediction.
⭐ Auto-saves every prediction to data/new_incoming/new_data.csv
"""

import os
import sys
import pandas as pd
from fastapi import APIRouter, Request
from api.schemas.churn_schema import ChurnPredictionInput, ChurnPredictionOutput
from api.middleware.rate_limiter import limiter
from api.utils.model_loader import get_model, get_transformer, get_label_encoder, get_config
from src.features.feature_engineering import FeatureEngineering
from src.utils.logger import get_logger
from src.utils.exception import ChurnModelException

logger = get_logger(__name__, log_file="api_requests.log")
router = APIRouter()


def _auto_save_prediction(input_data: dict, prediction: str):
    """
    ⭐ Auto-save every prediction + input to new_data.csv.
    This feeds the retraining loop — no human intervention needed.
    """
    try:
        config = get_config()
        save_path = config["data"]["new_incoming_path"]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        row = input_data.copy()
        row["Churn"] = prediction

        df_new = pd.DataFrame([row])

        # Append if file exists, create if not
        if os.path.exists(save_path):
            df_new.to_csv(save_path, mode="a", header=False, index=False)
        else:
            df_new.to_csv(save_path, mode="w", header=True, index=False)

        logger.info(f"Prediction auto-saved → {save_path}")

    except Exception as e:
        # Don't fail the API request if auto-save fails
        logger.error(f"Auto-save failed: {e}")


@router.post("/predict", response_model=ChurnPredictionOutput)
@limiter.limit("100/minute")
def predict(request: Request, payload: ChurnPredictionInput):
    """
    POST /predict
    Accepts customer data and returns churn prediction.
    Requires X-API-Key header for authentication.
    """
    try:
        logger.info(f"Prediction request | customerID={payload.customerID}")

        # Convert input to DataFrame
        input_dict = payload.model_dump()
        df = pd.DataFrame([input_dict])

        # Feature engineering
        fe = FeatureEngineering()
        df_engineered = fe.engineer(df)

        # Drop ID and target columns
        config = get_config()
        drop_cols = [
            config["model"]["customer_id_column"],
            config["model"]["target_column"]
        ]
        drop_cols = [c for c in drop_cols if c in df_engineered.columns]
        X = df_engineered.drop(columns=drop_cols)

        # Transform
        transformer = get_transformer()
        X_transformed = transformer.transform(X)

        # Predict
        model = get_model()
        label_encoder = get_label_encoder()

        prediction = model.predict(X_transformed)
        probability = model.predict_proba(X_transformed)[:, 1]
        label = label_encoder.inverse_transform(prediction)[0]

        # Auto-save prediction
        _auto_save_prediction(input_dict, label)

        message = (
            "High churn risk! Consider retention offer."
            if label == "Yes"
            else "Low churn risk. Customer likely to stay."
        )

        logger.info(
            f"Prediction complete | customerID={payload.customerID} "
            f"| result={label} | probability={probability[0]:.4f}"
        )

        return ChurnPredictionOutput(
            customerID=payload.customerID,
            prediction=label,
            churn_probability=round(float(probability[0]), 4),
            will_churn=bool(prediction[0] == 1),
            message=message
        )

    except Exception as e:
        raise ChurnModelException(e, sys) from e