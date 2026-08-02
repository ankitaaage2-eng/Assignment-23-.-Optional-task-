
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Ford Car ML Models",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Ford Car Machine Learning Prediction")
st.write("Frontend for Linear Regression, Logistic Regression, KNN and Naive Bayes")

# Load models
linear_model = joblib.load("linear_regression_model.pkl")
logistic_model = joblib.load("logistic_regression_model.pkl")
knn_model = joblib.load("knn_model.pkl")
naive_bayes_model = joblib.load("naive_bayes_model.pkl")

# Sidebar
st.sidebar.header("Select Model")

model_choice = st.sidebar.selectbox(
    "Choose a Model",
    [
        "Linear Regression",
        "Logistic Regression",
        "KNN",
        "Naive Bayes",
        "Model Comparison"
    ]
)

# Input fields
st.subheader("Enter Car Details")

model = st.selectbox(
    "Car Model",
    ["Fiesta", "Focus", "Puma", "Kuga", "EcoSport", "Other"]
)

year = st.number_input(
    "Year",
    min_value=1990,
    max_value=2026,
    value=2019
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic", "Semi-Auto"]
)

mileage = st.number_input(
    "Mileage",
    min_value=0,
    value=10000
)

fuelType = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "Hybrid", "Other"]
)

tax = st.number_input(
    "Tax",
    min_value=0,
    value=150
)

mpg = st.number_input(
    "MPG",
    min_value=0.0,
    value=50.0
)

engineSize = st.number_input(
    "Engine Size",
    min_value=0.1,
    value=1.0
)

input_data = pd.DataFrame({
    "model": [model],
    "year": [year],
    "transmission": [transmission],
    "mileage": [mileage],
    "fuelType": [fuelType],
    "tax": [tax],
    "mpg": [mpg],
    "engineSize": [engineSize]
})

# Prediction
if st.button("Predict"):

    if model_choice == "Linear Regression":

        prediction = linear_model.predict(input_data)[0]

        st.success(
            f"Predicted Car Price: ₹{prediction:,.2f}"
        )

    elif model_choice == "Logistic Regression":

        prediction = logistic_model.predict(input_data)[0]

        if prediction == 1:
            st.success("Prediction: Higher Price Category")
        else:
            st.info("Prediction: Lower Price Category")

    elif model_choice == "KNN":

        prediction = knn_model.predict(input_data)[0]

        if prediction == 1:
            st.success("Prediction: Higher Price Category")
        else:
            st.info("Prediction: Lower Price Category")

    elif model_choice == "Naive Bayes":

        prediction = naive_bayes_model.predict(input_data)[0]

        if prediction == 1:
            st.success("Prediction: Higher Price Category")
        else:
            st.info("Prediction: Lower Price Category")

    elif model_choice == "Model Comparison":

        st.subheader("Model Performance")

        comparison = pd.DataFrame({
            "Model": [
                "Linear Regression",
                "Logistic Regression",
                "KNN",
                "Naive Bayes"
            ],
            "Score": [
                0.845842,
                0.900668,
                0.924597,
                0.680000
            ]
        })

        st.table(comparison)

        st.success(
            "Best Classification Model: KNN (92.46% Accuracy)"
        )
