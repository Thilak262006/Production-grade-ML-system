"""
Model Configuration — defines all models and their hyperparameter grids.
Training pipeline reads from here — no hardcoded values anywhere.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def get_models() -> dict:
    """
    Returns a dictionary of model name → model instance.
    All models use random_state=42 for reproducibility.
    """
    return {
        "LogisticRegression": LogisticRegression(
            random_state=42, max_iter=500
        ),
        "RandomForest": RandomForestClassifier(
            random_state=42, class_weight="balanced"
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            verbosity=0
        ),
        "LightGBM": LGBMClassifier(
            random_state=42,
            class_weight="balanced",
            verbose=-1
        ),
    }


def get_param_grids() -> dict:
    """
    Returns hyperparameter grids for GridSearchCV.
    Keys must match get_models() keys exactly.
    """
    return {
        "LogisticRegression": {
            "C": [0.1, 1, 10],
            "solver": ["lbfgs", "liblinear"],
        },
        "RandomForest": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
        },
        "GradientBoosting": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5],
        },
        "XGBoost": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5],
        },
        "LightGBM": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "num_leaves": [31, 50],
        },
    }