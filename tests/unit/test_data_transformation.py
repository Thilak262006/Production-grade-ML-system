import pytest
import pandas as pd
from src.data.data_ingestion import DataIngestion
from src.data.data_splitting import DataSplitting
from src.features.feature_engineering import FeatureEngineering
from src.features.data_transformation import DataTransformation


@pytest.fixture
def sample_data():
    ingestion = DataIngestion()
    df = ingestion.load_data()
    splitting = DataSplitting()
    train, test, val = splitting.split(df)
    fe = FeatureEngineering()
    return fe.engineer(train), fe.engineer(test)


def test_fit_transform_returns_correct_shape(sample_data):
    train_fe, _ = sample_data
    dt = DataTransformation()
    X_train, y_train, transformer, le = dt.fit_transform(train_fe)
    assert X_train.shape[0] == len(train_fe)
    assert len(y_train) == len(train_fe)


def test_transform_only_matches_fit_shape(sample_data):
    train_fe, test_fe = sample_data
    dt = DataTransformation()
    X_train, y_train, transformer, le = dt.fit_transform(train_fe)
    X_test, y_test = dt.transform_only(test_fe, transformer)
    assert X_train.shape[1] == X_test.shape[1]


def test_label_encoder_values(sample_data):
    train_fe, _ = sample_data
    dt = DataTransformation()
    X_train, y_train, transformer, le = dt.fit_transform(train_fe)
    assert set(y_train).issubset({0, 1})