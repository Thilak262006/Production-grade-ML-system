from src.data.data_ingestion import DataIngestion
from src.data.data_splitting import DataSplitting
from src.features.feature_engineering import FeatureEngineering
from src.features.data_transformation import DataTransformation

# Step 1 - Load & split
ingestion = DataIngestion()
df = ingestion.load_data()

splitting = DataSplitting()
train, test, val = splitting.split(df)

# Step 2 - Feature engineering
fe = FeatureEngineering()
train = fe.engineer(train)
test  = fe.engineer(test)
val   = fe.engineer(val)

print(f"New columns: {[c for c in train.columns if c not in df.columns]}")

# Step 3 - Transformation
dt = DataTransformation()

# Fit on train only
X_train, y_train, transformer, le = dt.fit_transform(train)

# Transform test and val using fitted transformer
X_test,  y_test  = dt.transform_only(test, transformer)
X_val,   y_val   = dt.transform_only(val,  transformer)

print("SUCCESS — Phase 3 complete!")
print(f"X_train: {X_train.shape} | X_test: {X_test.shape} | X_val: {X_val.shape}")