"""
Model Training — trains all models and logs every run to MLflow.
Every experiment is tracked: params, metrics, model file.
"""

import sys
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score

from src.utils.logger import get_logger
from src.utils.exception import ChurnModelException
from src.utils.common import read_yaml, save_object
from src.models.model_config import get_models

logger = get_logger(__name__, log_file="training.log")


class ModelTraining:
    def __init__(self, config_path: str = "configs/config.yaml"):
        try:
            self.config = read_yaml(config_path)
            mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
            mlflow.set_experiment(self.config["mlflow"]["experiment_name"])
            logger.info(f"MLflow tracking URI: {self.config['mlflow']['tracking_uri']}")
        except Exception as e:
            raise ChurnModelException(e, sys) from e

    def train_all(self, X_train, y_train, X_val, y_val) -> dict:
        """
        Train every model in model_config.py.
        Log each run to MLflow with params + metrics.
        Returns a dict of model_name → {model, roc_auc}
        """
        try:
            models = get_models()
            results = {}

            for name, model in models.items():
                logger.info(f"Training {name}...")

                with mlflow.start_run(run_name=name):
                    # Train
                    model.fit(X_train, y_train)

                    # Evaluate on validation set
                    y_pred = model.predict(X_val)
                    y_prob = model.predict_proba(X_val)[:, 1]

                    acc = accuracy_score(y_val, y_pred)
                    f1 = f1_score(y_val, y_pred)
                    roc_auc = roc_auc_score(y_val, y_prob)

                    # Cross validation score
                    cv_scores = cross_val_score(
                        model, X_train, y_train,
                        cv=3, scoring="roc_auc", n_jobs=-1
                    )
                    cv_mean = cv_scores.mean()

                    # Log to MLflow
                    mlflow.log_param("model_name", name)
                    mlflow.log_metric("accuracy", acc)
                    mlflow.log_metric("f1_score", f1)
                    mlflow.log_metric("roc_auc", roc_auc)
                    mlflow.log_metric("cv_roc_auc_mean", cv_mean)
                    mlflow.sklearn.log_model(model, artifact_path="model")

                    logger.info(
                        f"{name} → acc={acc:.4f} | "
                        f"f1={f1:.4f} | roc_auc={roc_auc:.4f} | "
                        f"cv_auc={cv_mean:.4f}"
                    )

                    results[name] = {
                        "model": model,
                        "roc_auc": roc_auc,
                        "f1_score": f1,
                        "accuracy": acc,
                    }

            return results

        except Exception as e:
            raise ChurnModelException(e, sys) from e

    def get_best_model(self, results: dict):
        """Pick the model with highest roc_auc score."""
        try:
            best_name = max(results, key=lambda k: results[k]["roc_auc"])
            best = results[best_name]
            logger.info(
                f"Best model: {best_name} → roc_auc={best['roc_auc']:.4f}"
            )
            save_object(self.config["artifacts"]["model_path"], best["model"])
            logger.info(f"Best model saved → {self.config['artifacts']['model_path']}")
            return best_name, best["model"]
        except Exception as e:
            raise ChurnModelException(e, sys) from e
        