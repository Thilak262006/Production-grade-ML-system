from src.data.data_ingestion import DataIngestion
from src.data.data_splitting import DataSplitting
from src.features.feature_engineering import FeatureEngineering
from src.features.data_transformation import DataTransformation
from src.monitoring.data_drift_detection import DataDriftDetector
from src.monitoring.model_drift_detection import ModelDriftDetector
from src.utils.common import load_object

# Load data
ingestion = DataIngestion()
df = ingestion.load_data()

splitting = DataSplitting()
train, test, val = splitting.split(df)

# Feature engineering
fe = FeatureEngineering()
train_fe = fe.engineer(train)
test_fe  = fe.engineer(test)

# Transformation
dt = DataTransformation()
X_train, y_train, transformer, le = dt.fit_transform(train_fe)
X_test, y_test = dt.transform_only(test_fe, transformer)

# ── Data Drift Detection ──────────────────────────────────────────────────────
print("\n" + "="*50)
print("DATA DRIFT DETECTION")
print("="*50)
detector = DataDriftDetector()
drift_result = detector.detect(train_fe, test_fe)
print(f"Drift detected    : {drift_result['drift_detected']}")
print(f"Drifted columns   : {drift_result['drifted_columns']}/{drift_result['total_columns']}")
print(f"Drift share       : {drift_result['drift_share']*100:.1f}%")
print(f"Report saved      : {drift_result['report_path']}")

# ── Model Drift Detection ─────────────────────────────────────────────────────
print("\n" + "="*50)
print("MODEL DRIFT DETECTION")
print("="*50)
model_detector = ModelDriftDetector()
model_result = model_detector.detect(X_test, y_test)
print(f"Accuracy          : {model_result['accuracy']}")
print(f"ROC-AUC           : {model_result['roc_auc']}")
print(f"Retrain needed    : {model_result['retrain_needed']}")

print("\nSUCCESS — Phase 9 complete!")