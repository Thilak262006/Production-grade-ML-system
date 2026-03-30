"""
Feature Engineering — creates new features from raw columns.
These are business-logic features that help the model understand
customer behavior better.
"""

import sys
import pandas as pd
from src.utils.logger import get_logger
from src.utils.exception import ChurnModelException

logger = get_logger(__name__, log_file="training.log")


class FeatureEngineering:

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Starting feature engineering...")
            df = df.copy()

            # Fix TotalCharges — raw data has spaces instead of 0
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            df["TotalCharges"] = df["TotalCharges"].fillna(0)

            # 1. Tenure group
            df["tenure_group"] = pd.cut(
                df["tenure"],
                bins=[0, 12, 24, 48, 60, float("inf")],
                labels=["0-12", "13-24", "25-48", "49-60", "60+"],
                right=True
            ).astype(str)

            # 2. Total number of services subscribed
            service_cols = [
                "PhoneService", "MultipleLines", "InternetService",
                "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"
            ]
            df["total_services"] = df[service_cols].apply(
                lambda row: sum(1 for v in row if v == "Yes"), axis=1
            )

            # 3. Has any streaming service
            df["has_streaming"] = (
                (df["StreamingTV"] == "Yes") | (df["StreamingMovies"] == "Yes")
            ).astype(int)

            # 4. Is new customer (tenure < 12 months)
            df["is_new_customer"] = (df["tenure"] < 12).astype(int)

            # 5. Average monthly spend
            df["avg_monthly_spend"] = df.apply(
                lambda row: row["TotalCharges"] / row["tenure"]
                if row["tenure"] > 0 else row["MonthlyCharges"],
                axis=1
            )

            logger.info(f"New features added: tenure_group, total_services, "
                       f"has_streaming, is_new_customer, avg_monthly_spend")
            logger.info(f"Shape after engineering: {df.shape}")
            return df

        except Exception as e:
            raise ChurnModelException(e, sys) from e