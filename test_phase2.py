from src.data.data_ingestion import DataIngestion
from src.data.data_validation import DataValidation
from src.data.data_splitting import DataSplitting

# Step 1 - Load
ingestion = DataIngestion()
df = ingestion.load_data()

# Step 2 - Validate
validation = DataValidation()
validation.validate(df)

# Step 3 - Split
splitting = DataSplitting()
train, test, val = splitting.split(df)

print("SUCCESS — Phase 2 complete!")
print(f"Train: {len(train)} | Test: {len(test)} | Val: {len(val)}")
