from src.data.data_ingestion import DataIngestion
from src.data.data_splitting import DataSplitting
from src.features.feature_engineering import FeatureEngineering
from src.features.data_transformation import DataTransformation
from src.models.model_training import ModelTraining
from src.models.model_tuning import ModelTuning

# Data pipeline
ingestion = DataIngestion()
df = ingestion.load_data()

splitting = DataSplitting()
train, test, val = splitting.split(df)

fe = FeatureEngineering()
train = fe.engineer(train)
test  = fe.engineer(test)
val   = fe.engineer(val)

dt = DataTransformation()
X_train, y_train, transformer, le = dt.fit_transform(train)
X_val,   y_val   = dt.transform_only(val, transformer)

# Train all models
trainer = ModelTraining()
results = trainer.train_all(X_train, y_train, X_val, y_val)

# Get best model
best_name, best_model = trainer.get_best_model(results)

# Tune best model
tuner = ModelTuning()
tuned_model = tuner.tune(best_model, best_name, X_train, y_train, X_val, y_val)

print(f"\nSUCCESS — Phase 4 complete!")
print(f"Best model: {best_name}")
print(f"Saved → artifacts/best_model.joblib")