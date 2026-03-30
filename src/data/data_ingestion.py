"""
Data Ingestion — loads raw CSV into a pandas DataFrame.
Validates the file exists, logs basic info, and returns it.
"""

import sys
import pandas as pd
from src.utils.logger import get_logger
from src.utils.exception import ChurnModelException
from src.utils.common import read_yaml

logger = get_logger(__name__, log_file="training.log")


class DataIngestion:
    def __init__(self, config_path: str = "configs/config.yaml"):
        try:
            self.config = read_yaml(config_path)
            self.raw_data_path = self.config["data"]["raw_data_path"]
        except Exception as e:
            raise ChurnModelException(e, sys) from e

    def load_data(self) -> pd.DataFrame:
        try:
            logger.info(f"Loading raw data from: {self.raw_data_path}")
            df = pd.read_csv(self.raw_data_path)
            logger.info(f"Data loaded → shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            logger.info(f"Target distribution:\n{df['Churn'].value_counts()}")
            return df
        except Exception as e:
            raise ChurnModelException(e, sys) from e