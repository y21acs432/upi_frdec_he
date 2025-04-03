import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths for datasets and saved models
DATASET_PATH = os.path.join(BASE_DIR, "data", "synta_upi_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "xgboost_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "saved_models", "scaler.pkl")
LABLE_PATH = os.path.join(BASE_DIR,"saved_models","label_encoders.pkl")
ENCRYPTED_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "encrypted_model.pkl")
CONTEXT_PATH = os.path.join(BASE_DIR, "saved_context", "encryption_context.bin")
