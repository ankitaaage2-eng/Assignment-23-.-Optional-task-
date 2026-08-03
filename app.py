
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Ford Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Ford Car Price Prediction")
st.write("Predict whether the car belongs to Lower Price or Higher Price category.")

# Load dataset
df = pd.read_csv("ford_car_dataset.csv")

# Load saved model files
model = joblib.load("best_classification_model.pkl")
scaler = joblib.load("best_classification_scaler.pkl")
columns = joblib.load("logistic_columns.pkl")

st.subheader("Enter Car Details")

year = st.number_input(
    "Year",
    min_value=int(df["year"].min()),
    max_value=int(df["year"].max()),
    value=2018
)

mileage = st.number_input(
    "Mileage",
    min_value=0,
    value=10000
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

engine_size = st.number_input(
    "Engine Size",
    min_value=0.0,
    value=1.0
)

model_name = st.selectbox(
    "Model",
    sorted(df["model"].unique())
)

transmission = st.selectbox(
    "Transmission",
    sorted(df["transmission"].unique())
)

fuel_type = st.selectbox(
    "Fuel Type",
    sorted(df["fuelType"].unique())
)

if st.button("Predict"):

    input_df = pd.DataFrame({
        "year": [year],
        "mileage": [mileage],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engine_size]
    })

    # Add encoded columns
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Model
    model_col = "model_" + model_name
    if model_col in input_df.columns:
        input_df[model_col] = 1

    # Transmission
    transmission_col = "transmission_" + transmission
    if transmission_col in input_df.columns:
        input_df[transmission_col] = 1

    # Fuel type
    fuel_col = "fuelType_" + fuel_type
    if fuel_col in input_df.columns:
        input_df[fuel_col] = 1

    # Arrange columns exactly like training
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("Prediction: Higher Price Category 🚗")
    else:
        st.info("Prediction: Lower Price Category 🚙")
