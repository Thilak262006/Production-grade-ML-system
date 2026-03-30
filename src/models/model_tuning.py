"""
Model Tuning — runs GridSearchCV on the best model.
Logs the tuned model to MLflow as a separate run.
"""

import sys
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from src.utils.logger import get_logger
from src.utils.exception import ChurnModelException
from src.utils.common import read_yaml, save_object
from src.models.model_config import get_param_grids

logger = get_logger(__name__, log_file="training.log")


class ModelTuning:
    def __init__(self, config_path: str = "configs/config.yaml"):
        try:
            self.config = read_yaml(config_path)
            mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
            mlflow.set_experiment(self.config["mlflow"]["experiment_name"])
        except Exception as e:
            raise ChurnModelException(e, sys) from e

    def tune(self, model, model_name: str, X_train, y_train, X_val, y_val):
        """
        Run GridSearchCV on the best model.
        Logs tuned model + best params to MLflow.
        """
        try:
            logger.info(f"Tuning {model_name} with GridSearchCV...")

            param_grids = get_param_grids()

            if model_name not in param_grids:
                logger.warning(f"No param grid for {model_name} — skipping tuning")
                return model

            param_grid = param_grids[model_name]

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=3,
                scoring="roc_auc",
                n_jobs=-1,
                verbose=1,
            )

            with mlflow.start_run(run_name=f"{model_name}_tuned"):
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_

                # Evaluate tuned model
                y_pred = best_model.predict(X_val)
                y_prob = best_model.predict_proba(X_val)[:, 1]

                acc     = accuracy_score(y_val, y_pred)
                f1      = f1_score(y_val, y_pred)
                roc_auc = roc_auc_score(y_val, y_prob)

                # Log best params and metrics
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("roc_auc", roc_auc)
                mlflow.log_param("model_name", f"{model_name}_tuned")
                mlflow.sklearn.log_model(best_model, artifact_path="model")

                logger.info(f"Best params : {grid_search.best_params_}")
                logger.info(
                    f"Tuned scores → acc={acc:.4f} | "
                    f"f1={f1:.4f} | roc_auc={roc_auc:.4f}"
                )

                # Save tuned model to artifacts/
                save_object(self.config["artifacts"]["model_path"], best_model)
                logger.info("Tuned model saved → artifacts/best_model.joblib")

            return best_model

        except Exception as e:
            raise ChurnModelException(e, sys) from e