import streamlit as st
from src.inference.predict import predict_credit_score

st.title("Credit Scoring Simulator")

income = st.number_input("Income")
age = st.number_input("Age")

if st.button("Predict"):

    data = {
        "income": income,
        "age": age
    }

    score = predict_credit_score(data)

    st.write("Credit score:", score)