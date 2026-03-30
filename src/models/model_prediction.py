"""
Model Prediction — loads the best model and generates predictions.
Used by both the evaluation pipeline and the API.
"""

import sys
import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.exception import ChurnModelException
from src.utils.common import read_yaml, load_object

logger = get_logger(__name__, log_file="training.log")


class ModelPredictor:
    def __init__(self, config_path: str = "configs/config.yaml"):
        try:
            self.config = read_yaml(config_path)
            self.model = None
            self.transformer = None
            self.label_encoder = None
        except Exception as e:
            raise ChurnModelException(e, sys) from e

    def load_artifacts(self):
        """Load model, transformer and label encoder from artifacts/"""
        try:
            self.model = load_object(
                self.config["artifacts"]["model_path"]
            )
            self.transformer = load_object(
                self.config["artifacts"]["transformer_path"]
            )
            self.label_encoder = load_object(
                self.config["artifacts"]["label_encoder_path"]
            )
            logger.info("All artifacts loaded successfully")
            return self
        except Exception as e:
            raise ChurnModelException(e, sys) from e

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Takes a raw DataFrame → applies feature engineering
        → transforms → predicts → returns result dict.

        Args:
            df: Raw input DataFrame (same format as training data)

        Returns:
            dict with prediction and probability
        """
        try:
            from src.features.feature_engineering import FeatureEngineering

            # Apply feature engineering
            fe = FeatureEngineering()
            df = fe.engineer(df)

            # Drop ID and target if present
            drop_cols = [
                self.config["model"]["customer_id_column"],
                self.config["model"]["target_column"]
            ]
            drop_cols = [c for c in drop_cols if c in df.columns]
            X = df.drop(columns=drop_cols)

            # Transform
            X_transformed = self.transformer.transform(X)

            # Predict
            prediction = self.model.predict(X_transformed)
            probability = self.model.predict_proba(X_transformed)[:, 1]

            # Decode label
            label = self.label_encoder.inverse_transform(prediction)[0]

            result = {
                "prediction": label,
                "churn_probability": round(float(probability[0]), 4),
                "will_churn": bool(prediction[0] == 1)
            }

            logger.info(f"Prediction: {label} | Probability: {probability[0]:.4f}")
            return result

        except Exception as e:
            raise ChurnModelException(e, sys) from e