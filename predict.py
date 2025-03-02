import os
import pickle
import pandas as pd
import numpy as np
import logging
from model import PrivateXGBoostModel
from config import CONTEXT_PATH, DATASET_PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_feature_names():
    """Load dataset to get feature names (ensure the input transaction structure matches training data)."""
    try:
        if os.path.exists(DATASET_PATH):
            df = pd.read_csv(DATASET_PATH)
            return df.drop('is_fraud', axis=1).columns.tolist()
        else:
            logger.error(f"Dataset file not found: {DATASET_PATH}")
            raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")
    except Exception as e:
        logger.error(f"Error loading feature names: {e}")
        raise

def get_user_transaction(feature_names):
    """Take user input for transaction features and return a properly formatted DataFrame."""
    try:
        print("\nEnter transaction details:")

        # Get user input for each feature
        user_input = []
        for feature in feature_names:
            value = float(input(f"Enter value for {feature}: "))
            user_input.append(value)

        # Convert to DataFrame with correct feature names
        user_transaction_df = pd.DataFrame([user_input], columns=feature_names)
        return user_transaction_df
    except Exception as e:
        logger.error(f"Error getting user transaction: {e}")
        raise

def main():
    try:
        # Load feature names from dataset
        feature_names = load_feature_names()

        # Get transaction input from user
        user_transaction = get_user_transaction(feature_names)

        # Load the encrypted model
        private_model = PrivateXGBoostModel()
        private_model.load_model()  # Load encrypted weights and context

        # Encrypt user transaction
        encrypted_transaction = private_model.encrypt_transaction(user_transaction)

        # Perform encrypted prediction
        encrypted_prediction = private_model.predict_encrypted(encrypted_transaction)

        # Decrypt the prediction result
        final_prediction = private_model.decrypt_prediction(encrypted_prediction)

        # Print the prediction result
        print("\n🔍 Prediction Result:")
        print(f"Fraud Probability: {final_prediction:.4f}")
        print("Transaction is likely fraudulent!" if final_prediction > 0.5 else "Transaction is safe.")

    except Exception as e:
        logger.error(f"Error in prediction process: {e}")
        raise

if __name__ == "__main__":
    main()
