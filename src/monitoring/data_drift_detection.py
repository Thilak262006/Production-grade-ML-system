"""
Data Drift Detection — checks if incoming data distribution
has changed compared to training data.

Uses scipy KS-test (Kolmogorov-Smirnov) instead of Evidently
for Python 3.14 compatibility.

What is KS-test?
→ Compares two distributions statistically
→ p-value < 0.05 = drift detected
→ p-value > 0.05 = no drift
"""

import sys
import json
import pandas as pd
import numpy as np
from scipy import stats

from src.utils.logger import get_logger
from src.utils.exception import ChurnModelException
from src.utils.common import read_yaml, ensure_dir, save_json

logger = get_logger(__name__, log_file="monitoring.log")

# If more than 30% of columns drift → flag dataset drift
DRIFT_THRESHOLD = 0.30
# p-value threshold for KS test
P_VALUE_THRESHOLD = 0.05


class DataDriftDetector:
    def __init__(self, config_path: str = "configs/config.yaml"):
        try:
            self.config = read_yaml(config_path)
            ensure_dir(self.config["reports"]["dir"])
        except Exception as e:
            raise ChurnModelException(e, sys) from e

    def _ks_test(self, ref_col: pd.Series,
                 cur_col: pd.Series) -> dict:
        """Run KS test on a single column."""
        try:
            ref_clean = ref_col.dropna()
            cur_clean = cur_col.dropna()

            if len(ref_clean) == 0 or len(cur_clean) == 0:
                return {"drifted": False, "p_value": 1.0, "statistic": 0.0}

            # For categorical columns → compare value distributions
            if ref_col.dtype == object or str(ref_col.dtype) == "category":
                ref_dist = ref_clean.value_counts(normalize=True)
                cur_dist = cur_clean.value_counts(normalize=True)
                all_cats = set(ref_dist.index) | set(cur_dist.index)
                ref_vals = [ref_dist.get(c, 0) for c in all_cats]
                cur_vals = [cur_dist.get(c, 0) for c in all_cats]
                statistic, p_value = stats.ks_2samp(ref_vals, cur_vals)
            else:
                # For numerical columns → direct KS test
                statistic, p_value = stats.ks_2samp(
                    ref_clean.values, cur_clean.values
                )

            drifted = p_value < P_VALUE_THRESHOLD
            return {
                "drifted"  : drifted,
                "p_value"  : round(float(p_value), 4),
                "statistic": round(float(statistic), 4),
            }
        except Exception:
            return {"drifted": False, "p_value": 1.0, "statistic": 0.0}

    def detect(self, reference_df: pd.DataFrame,
               current_df: pd.DataFrame) -> dict:
        """
        Compare reference (training) data vs current (new) data.
        Runs KS test on every column.

        Args:
            reference_df: Training data
            current_df:   New incoming data

        Returns:
            dict with drift results per column
        """
        try:
            logger.info("Running data drift detection (KS-test)...")
            logger.info(f"Reference shape: {reference_df.shape}")
            logger.info(f"Current shape  : {current_df.shape}")

            # Drop target and ID columns
            target = self.config["model"]["target_column"]
            id_col = self.config["model"]["customer_id_column"]
            drop_cols = [c for c in [target, id_col]
                        if c in reference_df.columns]

            ref = reference_df.drop(columns=drop_cols, errors="ignore")
            cur = current_df.drop(columns=drop_cols, errors="ignore")

            # Keep only common columns
            common_cols = list(set(ref.columns) & set(cur.columns))
            ref = ref[common_cols]
            cur = cur[common_cols]

            # Run KS test on every column
            column_results = {}
            drifted_cols = []

            for col in common_cols:
                result = self._ks_test(ref[col], cur[col])
                column_results[col] = result
                if result["drifted"]:
                    drifted_cols.append(col)
                    logger.warning(
                        f"DRIFT in '{col}' | "
                        f"p_value={result['p_value']} | "
                        f"statistic={result['statistic']}"
                    )

            drift_share = len(drifted_cols) / len(common_cols)
            dataset_drift = drift_share > DRIFT_THRESHOLD

            summary = {
                "drift_detected"   : bool(dataset_drift),
                "drift_share"      : round(drift_share, 4),
                "drifted_columns"  : int(len(drifted_cols)),
                "total_columns"    : int(len(common_cols)),
                "drifted_col_names": drifted_cols,
                "column_results"   : {
                    col: {
                        "drifted"  : bool(v["drifted"]),
                        "p_value"  : float(v["p_value"]),
                        "statistic": float(v["statistic"]),
                    }
                    for col, v in column_results.items()
                },
                "report_path"      : "reports/data_drift_results.json",
            }

            # Save results
            save_json("reports/data_drift_results.json", summary)

            if dataset_drift:
                logger.warning(
                    f"DATASET DRIFT DETECTED! "
                    f"{len(drifted_cols)}/{len(common_cols)} columns drifted "
                    f"({drift_share*100:.1f}%)"
                )
            else:
                logger.info(
                    f"No significant drift. "
                    f"{len(drifted_cols)}/{len(common_cols)} columns drifted "
                    f"({drift_share*100:.1f}%)"
                )

            return summary

        except Exception as e:
            raise ChurnModelException(e, sys) from e