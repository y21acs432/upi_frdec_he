import xgboost as xgb
import tenseal as ts
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
import os
from encryption import ModelEncryption
from config import MODEL_PATH, SCALER_PATH, ENCRYPTED_MODEL_PATH, CONTEXT_PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrivateXGBoostModel:
    def __init__(self):
        self.model = None
        self.encryptor = ModelEncryption()
        self.scaler = StandardScaler()
        self.encrypted_weights = None
        self.encrypted_bias = None

    def train_model(self, X, y):
        """Train the XGBoost model with scaled data and handle class imbalance."""
        try:
            logger.info("Scaling training data...")
            #X_scaled = self.scaler.fit_transform(X)

            logger.info("Initializing XGBoost classifier...")
            self.model = xgb.XGBClassifier(
                n_estimators=500,  # More trees for better learning
                max_depth=3,  # Prevent overfitting
                learning_rate=0.05,  # Lower learning rate for better convergence
                subsample=0.8,  # Use 80% of data per tree
                colsample_bytree=0.8,  # Prevent overfitting
                scale_pos_weight=5, # Adjust weight for fraud cases (95%/5%)
                reg_alpha=0.1,
                reg_lambda=1.0
            )

            logger.info("Training the model...")
            self.model.fit(X, y)

            logger.info("Model training completed successfully.")
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise

    def encrypt_model_weights(self):
        """Encrypts the model weights using Homomorphic Encryption (CKKS)."""
        try:
            if self.model is None:
                raise ValueError("Model is not trained. Train the model before encrypting weights.")

            logger.info("Extracting model feature importances...")
            weights = self.model.feature_importances_.tolist()

            logger.info("Encrypting model weights using TenSEAL...")
            self.encrypted_weights = ts.ckks_vector(self.encryptor.context, weights)
            self.encrypted_bias = ts.ckks_vector(self.encryptor.context, [0.0])  # Default bias

            logger.info("Model weights encrypted successfully.")
        except Exception as e:
            logger.error(f"Error encrypting model weights: {e}")
            raise

    def save_model(self):
        """Saves the trained model, scaler, and encrypted parameters."""
        try:
            if self.model is None:
                raise ValueError("No trained model found. Train the model before saving.")

            logger.info("Saving the trained model and scaler...")
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(self.model, f)
            #with open(SCALER_PATH, 'wb') as f:
                #pickle.dump(self.scaler, f)

            logger.info("Saving encrypted model weights...")
            with open(ENCRYPTED_MODEL_PATH, 'wb') as f:
                pickle.dump((self.encrypted_weights.serialize(), self.encrypted_bias.serialize()), f)

            self.encryptor.save_context(CONTEXT_PATH)
            logger.info("Model, scaler, and encrypted parameters saved successfully.")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self):
        """Loads the trained model and its encrypted parameters."""
        try:
            logger.info("Loading trained model and scaler...")
            with open(MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            #with open(SCALER_PATH, 'rb') as f:
                #self.scaler = pickle.load(f)

            logger.info("Loading encryption context...")
            self.encryptor.load_context(CONTEXT_PATH)

            logger.info("Loading encrypted model weights...")
            with open(ENCRYPTED_MODEL_PATH, 'rb') as f:
                encrypted_data = pickle.load(f)

            self.encrypted_weights = ts.ckks_vector_from(self.encryptor.context, encrypted_data[0])
            self.encrypted_bias = ts.ckks_vector_from(self.encryptor.context, encrypted_data[1])

            logger.info("Model and encrypted parameters loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def encrypt_transaction(self, transaction):
        """Encrypts a transaction input using CKKS before prediction."""
        try:
            if self.model is None:
                raise ValueError("Model not loaded. Load the model before encrypting transactions.")

            logger.info("Scaling transaction data before encryption...")
            transaction = np.array(transaction).reshape(1,-1)
            scaled_transaction = self.scaler.fit_transform(transaction)

            logger.info("Encrypting transaction data...")
            return ts.ckks_vector(self.encryptor.context, scaled_transaction.flatten().tolist())
        except Exception as e:
            logger.error(f"Error encrypting transaction: {e}")
            raise

    def predict_encrypted(self, encrypted_transaction):
        """Performs encrypted prediction using dot product of encrypted weights and input."""
        try:
            if self.encrypted_weights is None:
                raise ValueError("Encrypted weights not available. Encrypt model weights before prediction.")

            logger.info("Performing encrypted prediction...")
            prediction = encrypted_transaction.dot(self.encrypted_weights) + self.encrypted_bias
            return prediction
        except Exception as e:
            logger.error(f"Error in encrypted prediction: {e}")
            raise

    def decrypt_prediction(self, encrypted_prediction):
        """Decrypts the encrypted model prediction and applies sigmoid for classification."""
        try:
            logger.info("Decrypting prediction output...")
            decrypted_value = encrypted_prediction.decrypt()[0]
            prediction_score = 1 / (1 + np.exp(-decrypted_value)) 

            logger.info(f"Decrypted Prediction Score: {prediction_score:.4f}")
            return prediction_score
        except Exception as e:
            logger.error(f"Error decrypting prediction: {e}")
            raise
