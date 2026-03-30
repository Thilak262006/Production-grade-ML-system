"""
Data Validation — checks the raw DataFrame against schema.yaml.
Fails loudly if data doesn't match expected schema.
"""

import sys
import pandas as pd
from src.utils.logger import get_logger
from src.utils.exception import ChurnModelException
from src.utils.common import read_yaml

logger = get_logger(__name__, log_file="training.log")


class DataValidation:
    def __init__(self, config_path: str = "configs/config.yaml",
                 schema_path: str = "configs/schema.yaml"):
        try:
            self.config = read_yaml(config_path)
            self.schema = read_yaml(schema_path)
        except Exception as e:
            raise ChurnModelException(e, sys) from e

    def validate(self, df: pd.DataFrame) -> bool:
        try:
            logger.info("Starting data validation...")
            errors = []

            # 1. Check minimum rows
            min_rows = self.schema["validation"]["min_rows"]
            if len(df) < min_rows:
                errors.append(f"Dataset has {len(df)} rows — minimum is {min_rows}")

            # 2. Check all required columns exist
            expected_cols = list(self.schema["columns"].keys())
            missing_cols = [c for c in expected_cols if c not in df.columns]
            if missing_cols:
                errors.append(f"Missing columns: {missing_cols}")

            # 3. Check null percentages
            max_null_pct = self.schema["validation"]["max_null_percentage"]
            for col in df.columns:
                null_pct = df[col].isnull().mean() * 100
                if null_pct > max_null_pct:
                    errors.append(f"Column '{col}' has {null_pct:.1f}% nulls (max: {max_null_pct}%)")

            # 4. Check allowed values for categorical columns
            for col, rules in self.schema["columns"].items():
                if col not in df.columns:
                    continue
                if "allowed_values" in rules:
                    unique_vals = df[col].dropna().unique().tolist()
                    invalid = [v for v in unique_vals if v not in rules["allowed_values"]]
                    if invalid:
                        errors.append(f"Column '{col}' has invalid values: {invalid}")

            # Report results
            if errors:
                for err in errors:
                    logger.error(f"Validation error: {err}")
                raise ValueError(f"Data validation failed with {len(errors)} error(s)")

            logger.info("Data validation passed!")
            return True

        except Exception as e:
            raise ChurnModelException(e, sys) from e