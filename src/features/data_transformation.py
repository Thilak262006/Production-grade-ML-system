"""
Data Transformation — scales numerical features, encodes categoricals.
Saves the fitted transformer to artifacts/ so the API can use it later.
"""

import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from src.utils.logger import get_logger
from src.utils.exception import ChurnModelException
from src.utils.common import read_yaml, save_object, ensure_dir

logger = get_logger(__name__, log_file="training.log")


class DataTransformation:
    def __init__(self, config_path: str = "configs/config.yaml"):
        try:
            self.config = read_yaml(config_path)
            self.target = self.config["model"]["target_column"]
            self.id_col = self.config["model"]["customer_id_column"]
            ensure_dir(self.config["artifacts"]["dir"])
        except Exception as e:
            raise ChurnModelException(e, sys) from e

    def _get_feature_groups(self, df: pd.DataFrame):
        """Identify numerical and categorical columns automatically."""
        drop_cols = [self.target, self.id_col]
        feature_cols = [c for c in df.columns if c not in drop_cols]

        numerical_cols = df[feature_cols].select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        categorical_cols = df[feature_cols].select_dtypes(
            include=["object"]
        ).columns.tolist()

        return numerical_cols, categorical_cols

    def build_transformer(self, df: pd.DataFrame) -> ColumnTransformer:
        """Build a sklearn ColumnTransformer pipeline."""
        try:
            numerical_cols, categorical_cols = self._get_feature_groups(df)

            logger.info(f"Numerical columns  : {numerical_cols}")
            logger.info(f"Categorical columns: {categorical_cols}")

            numerical_pipeline = Pipeline(steps=[
                ("scaler", StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])

            transformer = ColumnTransformer(transformers=[
                ("num", numerical_pipeline, numerical_cols),
                ("cat", categorical_pipeline, categorical_cols),
            ])

            return transformer, numerical_cols, categorical_cols

        except Exception as e:
            raise ChurnModelException(e, sys) from e

    def fit_transform(self, df: pd.DataFrame):
        """
        Fit transformer on training data and transform it.
        Saves the fitted transformer to artifacts/.
        """
        try:
            logger.info("Fitting data transformer on training data...")

            # Encode target: Yes → 1, No → 0
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(df[self.target])

            # Drop ID and target from features
            X = df.drop(columns=[self.target, self.id_col])

            transformer, numerical_cols, categorical_cols = self.build_transformer(df)

            X_transformed = transformer.fit_transform(X)

            # Save transformer and label encoder to artifacts/
            save_object(self.config["artifacts"]["transformer_path"], transformer)
            save_object(self.config["artifacts"]["label_encoder_path"], label_encoder)

            logger.info(f"Transformed shape: {X_transformed.shape}")
            logger.info("Transformer saved to artifacts/")

            return X_transformed, y, transformer, label_encoder

        except Exception as e:
            raise ChurnModelException(e, sys) from e

    def transform_only(self, df: pd.DataFrame, transformer: ColumnTransformer):
        """
        Transform test/val data using already-fitted transformer.
        Does NOT refit — uses the saved transformer from training.
        """
        try:
            logger.info("Transforming data with fitted transformer...")

            y = None
            if self.target in df.columns:
                from src.utils.common import load_object
                label_encoder = load_object(
                    self.config["artifacts"]["label_encoder_path"]
                )
                y = label_encoder.transform(df[self.target])

            drop_cols = [c for c in [self.target, self.id_col] if c in df.columns]
            X = df.drop(columns=drop_cols)
            X_transformed = transformer.transform(X)

            logger.info(f"Transformed shape: {X_transformed.shape}")
            return X_transformed, y

        except Exception as e:
            raise ChurnModelException(e, sys) from e