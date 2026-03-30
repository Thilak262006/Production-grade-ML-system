"""
Data Splitting — splits the raw DataFrame into train/test/val CSV files.
Saves them to data/processed/
"""

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger
from src.utils.exception import ChurnModelException
from src.utils.common import read_yaml, ensure_dir

logger = get_logger(__name__, log_file="training.log")


class DataSplitting:
    def __init__(self, config_path: str = "configs/config.yaml",
                 params_path: str = "configs/params.yaml"):
        try:
            self.config = read_yaml(config_path)
            self.params = read_yaml(params_path)
            self.target = self.config["model"]["target_column"]
            self.test_size = self.params["data_processing"]["test_size"]
            self.val_size = self.params["data_processing"]["val_size"]
            self.random_state = self.params["data_processing"]["random_state"]
        except Exception as e:
            raise ChurnModelException(e, sys) from e

    def split(self, df: pd.DataFrame):
        try:
            logger.info(f"Splitting data | test_size={self.test_size} | val_size={self.val_size}")

            # First split → train+val vs test
            train_val, test = train_test_split(
                df,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=df[self.target]
            )

            # Second split → train vs val
            val_ratio = self.val_size / (1 - self.test_size)
            train, val = train_test_split(
                train_val,
                test_size=val_ratio,
                random_state=self.random_state,
                stratify=train_val[self.target]
            )

            # Save to data/processed/
            ensure_dir(self.config["data"]["processed_dir"])
            train.to_csv(self.config["data"]["train_path"], index=False)
            test.to_csv(self.config["data"]["test_path"], index=False)
            val.to_csv(self.config["data"]["val_path"], index=False)

            logger.info(f"Train size : {len(train)}")
            logger.info(f"Test size  : {len(test)}")
            logger.info(f"Val size   : {len(val)}")
            logger.info(f"Saved → {self.config['data']['processed_dir']}")

            return train, test, val

        except Exception as e:
            raise ChurnModelException(e, sys) from e