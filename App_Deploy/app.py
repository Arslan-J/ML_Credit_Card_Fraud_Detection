import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load the trained model
lr = pickle.load(open("logreg.pkl", "rb"))

# Load the scaler (assuming it's saved as 'scaler.pkl')
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("💳 Credit Card Fraud Detection App")
st.write("Enter values for the 30 features to predict fraud (1 = fraud, 0 = not fraud).")

# Create a dictionary to collect input values
values = {}
for i in range(1, 29):  # Features V1 to V28
    values[f'V{i}'] = st.number_input(f"V{i}")

feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

Amount = st.number_input("Amount")

Time = st.number_input("Time")

# Collect all the inputs into an array
features = np.array([[Time, *values.values(), Amount]])

features_df = pd.DataFrame(features, columns=feature_names)

# Scale the input features
scaled_features = scaler.transform(features_df)

# Predict button
if st.button("Predict"):
    prediction = lr.predict(scaled_features)
    st.write("🧠 Prediction:", "⚠️ Fraud" if prediction[0] == 1 else "✅ Not Fraud")