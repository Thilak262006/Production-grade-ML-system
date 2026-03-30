"""
Model Evaluation — generates all evaluation metrics and report charts.
Saves confusion matrix, ROC curve, feature importance to reports/
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    roc_curve
)

from src.utils.logger import get_logger
from src.utils.exception import ChurnModelException
from src.utils.common import read_yaml, save_json, ensure_dir

logger = get_logger(__name__, log_file="training.log")


class ModelEvaluation:
    def __init__(self, config_path: str = "configs/config.yaml"):
        try:
            self.config = read_yaml(config_path)
            ensure_dir(self.config["reports"]["dir"])
        except Exception as e:
            raise ChurnModelException(e, sys) from e

    def evaluate(self, model, X_test, y_test) -> dict:
        """Compute all metrics and return as dictionary."""
        try:
            logger.info("Evaluating model on test set...")

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy"  : round(accuracy_score(y_test, y_pred), 4),
                "f1_score"  : round(f1_score(y_test, y_pred), 4),
                "roc_auc"   : round(roc_auc_score(y_test, y_prob), 4),
            }

            logger.info(f"Accuracy : {metrics['accuracy']}")
            logger.info(f"F1 Score : {metrics['f1_score']}")
            logger.info(f"ROC-AUC  : {metrics['roc_auc']}")
            logger.info(f"\n{classification_report(y_test, y_pred)}")

            # Save metrics to JSON
            save_json(self.config["reports"]["metrics_json"], metrics)

            return metrics, y_pred, y_prob

        except Exception as e:
            raise ChurnModelException(e, sys) from e

    def plot_confusion_matrix(self, y_test, y_pred):
        """Save confusion matrix chart to reports/"""
        try:
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))

            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            plt.colorbar(im, ax=ax)

            ax.set(
                xticks=[0, 1], yticks=[0, 1],
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"],
                xlabel="Predicted", ylabel="Actual",
                title="Confusion Matrix"
            )

            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]),
                            ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black",
                            fontsize=14)

            plt.tight_layout()
            plt.savefig(self.config["reports"]["confusion_matrix"])
            plt.close()
            logger.info(f"Confusion matrix saved → {self.config['reports']['confusion_matrix']}")

        except Exception as e:
            raise ChurnModelException(e, sys) from e

    def plot_roc_curve(self, y_test, y_prob):
        """Save ROC curve chart to reports/"""
        try:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(fpr, tpr, color="darkorange", lw=2,
                    label=f"ROC Curve (AUC = {auc:.4f})")
            ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--",
                    label="Random Classifier")
            ax.set(xlabel="False Positive Rate",
                   ylabel="True Positive Rate",
                   title="ROC Curve")
            ax.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(self.config["reports"]["roc_curve"])
            plt.close()
            logger.info(f"ROC curve saved → {self.config['reports']['roc_curve']}")

        except Exception as e:
            raise ChurnModelException(e, sys) from e

    def plot_feature_importance(self, model, feature_names: list):
        """Save feature importance chart to reports/"""
        try:
            if not hasattr(model, "feature_importances_"):
                logger.warning("Model does not support feature importance — skipping")
                return

            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]  # top 20

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(
                range(len(indices)),
                importances[indices],
                color="steelblue"
            )
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([feature_names[i] for i in indices])
            ax.invert_yaxis()
            ax.set(xlabel="Importance", title="Top 20 Feature Importances")
            plt.tight_layout()
            plt.savefig(self.config["reports"]["feature_importance"])
            plt.close()
            logger.info(f"Feature importance saved → {self.config['reports']['feature_importance']}")

        except Exception as e:
            raise ChurnModelException(e, sys) from e