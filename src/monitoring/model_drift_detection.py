"""
Model Drift Detection — checks if model accuracy is degrading over time.

What is model drift?
→ Model was 87% accurate at launch
→ 3 months later it's 71% accurate
→ Time to retrain!

How we detect it:
→ Compare current accuracy against a baseline threshold
→ If accuracy drops below threshold → trigger alert
"""

import sys
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score,
    roc_auc_score, classification_report
)

from src.utils.logger import get_logger
from src.utils.exception import ChurnModelException
from src.utils.common import read_yaml, load_object, save_json, ensure_dir

logger = get_logger(__name__, log_file="monitoring.log")

# Threshold below which we consider model is degrading
ACCURACY_THRESHOLD = 0.70
ROC_AUC_THRESHOLD  = 0.75


class ModelDriftDetector:
    def __init__(self, config_path: str = "configs/config.yaml"):
        try:
            self.config = read_yaml(config_path)
            ensure_dir(self.config["reports"]["dir"])
        except Exception as e:
            raise ChurnModelException(e, sys) from e

    def detect(self, X_current, y_current) -> dict:
        """
        Run model on current data and compare metrics against thresholds.

        Args:
            X_current: Transformed feature matrix of new data
            y_current: True labels of new data

        Returns:
            dict with drift results and whether retraining is needed
        """
        try:
            logger.info("Running model drift detection...")

            # Load current best model
            model = load_object(self.config["artifacts"]["model_path"])

            # Generate predictions
            y_pred = model.predict(X_current)
            y_prob = model.predict_proba(X_current)[:, 1]

            # Calculate metrics
            accuracy = round(accuracy_score(y_current, y_pred), 4)
            f1       = round(f1_score(y_current, y_pred), 4)
            roc_auc  = round(roc_auc_score(y_current, y_prob), 4)

            logger.info(f"Current accuracy : {accuracy}")
            logger.info(f"Current F1 score : {f1}")
            logger.info(f"Current ROC-AUC  : {roc_auc}")
            logger.info(f"\n{classification_report(y_current, y_pred)}")

            # Check against thresholds
            accuracy_drift = accuracy < ACCURACY_THRESHOLD
            roc_auc_drift  = roc_auc < ROC_AUC_THRESHOLD
            retrain_needed = accuracy_drift or roc_auc_drift

            result = {
                "accuracy"        : accuracy,
                "f1_score"        : f1,
                "roc_auc"         : roc_auc,
                "accuracy_drift"  : accuracy_drift,
                "roc_auc_drift"   : roc_auc_drift,
                "retrain_needed"  : retrain_needed,
                "accuracy_threshold": ACCURACY_THRESHOLD,
                "roc_auc_threshold" : ROC_AUC_THRESHOLD,
            }

            if retrain_needed:
                logger.warning(
                    f"MODEL DRIFT DETECTED! Retraining recommended. "
                    f"accuracy={accuracy} (threshold={ACCURACY_THRESHOLD}) | "
                    f"roc_auc={roc_auc} (threshold={ROC_AUC_THRESHOLD})"
                )
            else:
                logger.info(
                    f"Model performance is stable. "
                    f"accuracy={accuracy} | roc_auc={roc_auc}"
                )

            # Save results to reports/
            save_json("reports/model_drift_results.json", result)

            return result

        except Exception as e:
            raise ChurnModelException(e, sys) from e