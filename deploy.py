import streamlit as st
import pickle
import tenseal as ts
import numpy as np
import pandas as pd
from model import PrivateXGBoostModel
from sklearn.preprocessing import LabelEncoder

# Load the trained encrypted model
@st.cache_resource
def load_model():
    with open(r"C:\ppupifduhe\saved_models\encrypted_model.pkl", "rb") as file:
        model = pickle.load(file)
    
    if isinstance(model, tuple):  # Unpack if needed
        model = model[0]
    
    return model

# Load label encoders
@st.cache_resource
def load_label_encoders():
    with open(r"C:\ppupifduhe\saved_models\label_encoders.pkl", "rb") as file:
        label_encoders = pickle.load(file)
    return label_encoders

model = PrivateXGBoostModel
model.load_model()
label_encoders = load_label_encoders()

# CKKS Encryption Context
@st.cache_resource
def create_ckks_context():
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.generate_galois_keys()
    context.global_scale = 2**40
    return context

context = create_ckks_context()

# Input Fields
st.title("Encrypted Transaction Prediction")

amount = st.number_input("Transaction Amount", value=0.0)
location = st.text_input("Enter Location")
device_type = st.selectbox("Select Device Type", options=["Trusted", "Untrusted"])
merchant_category = st.selectbox("Select Merchant Category", options=["Grocery", "Electronics","Food","Fashion","Travel","Others"])
transaction_frequency = st.number_input("Transaction Frequency", value=0.0)
num_transactions_24hrs = st.number_input("Transactions in Last 24 Hours", value=0)
num_transactions_48hrs = st.number_input("Transactions in Last 48 Hours", value=0)
time_since_last_transaction = st.number_input("Time Since Last Transaction (minutes)", value=0.0)

# Label Encoding for categorical features
def encode_feature(feature, feature_name):
    if feature_name in label_encoders:
        return label_encoders[feature_name].transform([feature])[0]
    else:
        return 0  # Default encoding if not found

if st.button("Predict"):
    location_encoded = encode_feature(location, "location")
    device_type_encoded = encode_feature(device_type, "device_type")
    merchant_category_encoded = encode_feature(merchant_category, "merchant_category")

    # Prepare input data
    input_data = np.array([[amount, location_encoded, device_type_encoded, merchant_category_encoded,
                             transaction_frequency, num_transactions_24hrs, num_transactions_48hrs, 
                             time_since_last_transaction]], dtype=np.float32)

    # Encrypt input
    encrypted_input = ts.ckks_vector(context, input_data.flatten().tolist())

    # Ensure model has predict_encrypted method
    if hasattr(model, "predict_encrypted"):
        encrypted_output = model.predict_encrypted([encrypted_input])
        st.write("Encrypted Prediction:", encrypted_output)
    else:
        st.error("Error: The model does not have 'predict_encrypted()' method. Check the model class.")
