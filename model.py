import xgboost as xgb
import tenseal as ts
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
import os
from encryption import ModelEncryption
from config import MODEL_PATH, SCALER_PATH, ENCRYPTED_MODEL_PATH, CONTEXT_PATH

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
        try:
            X_scaled = self.scaler.fit_transform(X)
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                objective='binary:logistic',
                use_label_encoder=False,
                eval_metric='logloss'
            )
            self.model.fit(X_scaled, y)
            logger.info("Model training completed successfully")
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise

    def encrypt_model_weights(self):
        try:
            weights = self.model.feature_importances_.tolist()
            self.encrypted_weights = ts.ckks_vector(self.encryptor.context, weights)
            self.encrypted_bias = ts.ckks_vector(self.encryptor.context, [0.0])
            logger.info("Model weights encrypted successfully")
        except Exception as e:
            logger.error(f"Error encrypting model weights: {e}")
            raise

    def save_model(self):
        try:
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(self.model, f)
            with open(SCALER_PATH, 'wb') as f:
                pickle.dump(self.scaler, f)
            with open(ENCRYPTED_MODEL_PATH, 'wb') as f:
                pickle.dump((self.encrypted_weights.serialize(), self.encrypted_bias.serialize()), f)
            self.encryptor.save_context(CONTEXT_PATH)
            logger.info("Model, scaler, and encrypted parameters saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self):
        try:
            with open(MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            with open(SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
            self.encryptor.load_context(CONTEXT_PATH)
            with open(ENCRYPTED_MODEL_PATH, 'rb') as f:
                encrypted_data = pickle.load(f)
            self.encrypted_weights = ts.ckks_vector_from(self.encryptor.context, encrypted_data[0])
            self.encrypted_bias = ts.ckks_vector_from(self.encryptor.context, encrypted_data[1])
            logger.info("Model and encrypted parameters loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def encrypt_transaction(self, transaction):
        try:
            scaled_transaction = self.scaler.transform(transaction)
            return ts.ckks_vector(self.encryptor.context, scaled_transaction.flatten().tolist())
        except Exception as e:
            logger.error(f"Error encrypting transaction: {e}")
            raise
    
    def predict_encrypted(self, encrypted_transaction):
        try:
            prediction = encrypted_transaction.dot(self.encrypted_weights) + self.encrypted_bias
            return prediction
        except Exception as e:
            logger.error(f"Error in encrypted prediction: {e}")
            raise
    
    def decrypt_prediction(self, encrypted_prediction):
        try:
            decrypted_value = encrypted_prediction.decrypt()[0]
            return 1 / (1 + np.exp(-decrypted_value))
        except Exception as e:
            logger.error(f"Error decrypting prediction: {e}")
            raise