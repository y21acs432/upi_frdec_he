import os
import pickle
import pandas as pd
import numpy as np
import logging
from model import PrivateXGBoostModel
from config import CONTEXT_PATH, DATASET_PATH
from sklearn.preprocessing import LabelEncoder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load dataset for label encoding
df = pd.read_csv(DATASET_PATH)

# Initialize Label Encoders
location_encoder = LabelEncoder()
device_type_encoder = LabelEncoder()
merchant_category_encoder = LabelEncoder()

location_encoder.fit(df['location'])
device_type_encoder.fit(df['device_type'])
merchant_category_encoder.fit(df['merchant_category'])

def load_feature_names():
    """Load dataset to get feature names, excluding 'is_fraud', 'transaction_time', and 'upi_id'."""
    try:
        if os.path.exists(DATASET_PATH):
            df = pd.read_csv(DATASET_PATH)
            return df.drop(columns=['is_fraud', 'transaction_time', 'upi_id'], errors='ignore').columns.tolist()
        else:
            logger.error(f"Dataset file not found: {DATASET_PATH}")
            raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")
    except Exception as e:
        logger.error(f"Error loading feature names: {e}")
        raise

def get_user_transaction(feature_names):
    """Take user input for transaction details and return a properly formatted DataFrame."""
    try:
        print("\nEnter transaction details:")

        # Read transaction_time and upi_id (but do not use them for prediction)
        transaction_time = input("Enter transaction time: ")
        upi_id = input("Enter UPI ID: ")

        user_input = {}

        for feature in feature_names:
            value = input(f"Enter value for {feature}: ")

            # Convert categorical values using label encoders
            if feature == "location":
                value = location_encoder.transform([value])[0] if value in location_encoder.classes_ else -1
            elif feature == "device_type":
                value = device_type_encoder.transform([value])[0] if value in device_type_encoder.classes_ else -1
            elif feature == "merchant_category":
                value = merchant_category_encoder.transform([value])[0] if value in merchant_category_encoder.classes_ else -1
            else:
                value = float(value)  # Convert numerical inputs to float

            user_input[feature] = value

        user_transaction_df = pd.DataFrame([user_input])

        return transaction_time, upi_id, user_transaction_df

    except Exception as e:
        logger.error(f"Error getting user transaction: {e}")
        raise

def main():
    try:
        # Load feature names
        feature_names = load_feature_names()

        # Get user transaction input
        transaction_time, upi_id, user_transaction = get_user_transaction(feature_names)

        # Load encrypted model
        private_model = PrivateXGBoostModel()
        private_model.load_model()  # Load encrypted model and context

        # Encrypt transaction data
        encrypted_transaction = private_model.encrypt_transaction(user_transaction)

        # Make encrypted prediction
        encrypted_prediction = private_model.predict_encrypted(encrypted_transaction)

        # Decrypt and display result
        final_prediction = private_model.decrypt_prediction(encrypted_prediction)

        print("\n Prediction Result:")
        print(f"Transaction Time: {transaction_time}")
        print(f"UPI ID: {upi_id}")
        print(f"Fraud Probability: {final_prediction:.4f}")
        print("Fraudulent Transaction!" if final_prediction > 0.5 else "Safe Transaction.")

    except Exception as e:
        logger.error(f"Error in prediction process: {e}")
        raise

if __name__ == "__main__":
    main()
