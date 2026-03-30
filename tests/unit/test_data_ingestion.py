import pytest
import pandas as pd
from src.data.data_ingestion import DataIngestion


def test_load_data_returns_dataframe():
    ingestion = DataIngestion()
    df = ingestion.load_data()
    assert isinstance(df, pd.DataFrame)


def test_load_data_has_correct_columns():
    ingestion = DataIngestion()
    df = ingestion.load_data()
    assert "Churn" in df.columns
    assert "customerID" in df.columns


def test_load_data_has_correct_shape():
    ingestion = DataIngestion()
    df = ingestion.load_data()
    assert df.shape[0] > 0
    assert df.shape[1] == 21


def test_load_data_target_values():
    ingestion = DataIngestion()
    df = ingestion.load_data()
    assert set(df["Churn"].unique()) == {"Yes", "No"}