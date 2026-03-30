from src.data.data_ingestion import DataIngestion
from src.data.data_splitting import DataSplitting
from src.features.feature_engineering import FeatureEngineering
from src.features.data_transformation import DataTransformation
from src.models.model_prediction import ModelPredictor
from src.models.model_evaluation import ModelEvaluation
import pandas as pd

# Data pipeline
ingestion = DataIngestion()
df = ingestion.load_data()

splitting = DataSplitting()
train, test, val = splitting.split(df)

fe = FeatureEngineering()
test = fe.engineer(test)

dt = DataTransformation()
dt.fit_transform(fe.engineer(train))
X_test, y_test = dt.transform_only(test, dt.transformer if hasattr(dt, 'transformer') else __import__('src.utils.common', fromlist=['load_object']).load_object('artifacts/data_transformer.joblib'))

# Load artifacts
from src.utils.common import load_object
transformer = load_object("artifacts/data_transformer.joblib")
X_test, y_test = dt.transform_only(test, transformer)

# Evaluate
model = load_object("artifacts/best_model.joblib")
evaluator = ModelEvaluation()
metrics, y_pred, y_prob = evaluator.evaluate(model, X_test, y_test)

# Generate charts
evaluator.plot_confusion_matrix(y_test, y_pred)
evaluator.plot_roc_curve(y_test, y_prob)

# Feature importance
from sklearn.pipeline import Pipeline
feature_names = transformer.get_feature_names_out().tolist()
evaluator.plot_feature_importance(model, feature_names)

# Test single prediction
predictor = ModelPredictor()
predictor.load_artifacts()

sample = pd.DataFrame([{
    "customerID": "TEST-001",
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": "140.70",
    "Churn": "No"
}])

result = predictor.predict(sample)
print(f"\nSUCCESS — Phase 5 complete!")
print(f"Metrics: {metrics}")
print(f"Sample prediction: {result}")