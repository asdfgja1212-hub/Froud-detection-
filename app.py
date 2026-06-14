import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

model = tf.keras.models.load_model("fraud_detection_hybrid.keras")

with open("scaler.pkl","rb") as f:
    scaler = pickle.load(f)

st.title("Financial Fraud Detection")

type_value = st.selectbox(
    "Transaction Type",
    [0,1,2,3,4]
)

amount = st.number_input("Amount")

oldbalanceOrg = st.number_input("Old Balance Origin")

newbalanceOrig = st.number_input("New Balance Origin")

oldbalanceDest = st.number_input("Old Balance Destination")

newbalanceDest = st.number_input("New Balance Destination")

isFlaggedFraud = st.number_input("Flagged Fraud",0,1)

if st.button("Predict"):

    data = np.array([[
        type_value,
        amount,
        oldbalanceOrg,
        newbalanceOrig,
        oldbalanceDest,
        newbalanceDest,
        isFlaggedFraud
    ]])

    data = scaler.transform(data)

    data = data.reshape(
        data.shape[0],
        data.shape[1],
        1
    )

    prediction = model.predict(data)

    if prediction[0][0] > 0.5:
        st.error("Fraud Transaction")
    else:
        st.success("Legitimate Transaction")
