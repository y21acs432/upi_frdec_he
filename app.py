import streamlit as st
import pickle
import tenseal as ts
import numpy as np
import pandas as pd
from model import PrivateXGBoostModel
from predict import location_encoder, device_type_encoder, merchant_category_encoder, load_feature_names
from config import DATASET_PATH

# Load feature names
feature_names = load_feature_names()

# Load trained encrypted model
private_model = PrivateXGBoostModel()
private_model.load_model()

# Streamlit UI
st.title("Privacy-Preserving Fraud Detection")

st.write("Enter transaction details:")

user_input = {}

# Collect input from user
for feature in feature_names:
    if feature in ["location", "device_type", "merchant_category"]:
        value = st.text_input(f"Enter {feature}:")
        if feature == "location":
            value = location_encoder.transform([value])[0] if value in location_encoder.classes_ else -1
        elif feature == "device_type":
            value = device_type_encoder.transform([value])[0] if value in device_type_encoder.classes_ else -1
        elif feature == "merchant_category":
            value = merchant_category_encoder.transform([value])[0] if value in merchant_category_encoder.classes_ else -1
    else:
        value = st.number_input(f"Enter {feature}:", value=0.0)
    
    user_input[feature] = value

# Convert input to DataFrame
user_transaction_df = pd.DataFrame([user_input])

if st.button("Predict"):
    try:
        # Encrypt transaction data
        encrypted_transaction = private_model.encrypt_transaction(user_transaction_df)

        # Make encrypted prediction
        encrypted_prediction = private_model.predict_encrypted(encrypted_transaction)

        # Decrypt the result
        final_prediction = private_model.decrypt_prediction(encrypted_prediction)

        st.write("### Prediction Result")
        st.write(f"**Fraud Probability:** {final_prediction:.4f}")
        st.write("🚨 Fraudulent Transaction!" if final_prediction > 0.5 else "✅ Safe Transaction.")

    except Exception as e:
        st.error(f"Error: {e}")
