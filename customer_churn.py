import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load trained pipeline
pipeline = pickle.load(open("churn_pipeline_v3.pkl", "rb"))

st.title("📊 Customer Churn Prediction App")
st.write("Predict customer churn and see which features matter most.")

# ---- SIDEBAR INPUTS ----
st.sidebar.header("Customer Details")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", [1, 0])
partner = st.sidebar.selectbox("Partner", ["Yes","No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes","No"])
tenure = st.sidebar.number_input("Tenure (months)", 0, 100)
phone_service = st.sidebar.selectbox("Phone Service", ["Yes","No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No","Yes","No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL","Fiber optic","No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes","No","No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes","No","No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes","No","No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes","No","No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes","No","No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes","No","No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month","One year","Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes","No"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])
monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0)
total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0)

# ---- CREATE INPUT DATAFRAME ----
input_df = pd.DataFrame({
    "gender":[gender],
    "SeniorCitizen":[senior],
    "Partner":[partner],
    "Dependents":[dependents],
    "tenure":[tenure],
    "PhoneService":[phone_service],
    "MultipleLines":[multiple_lines],
    "InternetService":[internet_service],
    "OnlineSecurity":[online_security],
    "OnlineBackup":[online_backup],
    "DeviceProtection":[device_protection],
    "TechSupport":[tech_support],
    "StreamingTV":[streaming_tv],
    "StreamingMovies":[streaming_movies],
    "Contract":[contract],
    "PaperlessBilling":[paperless_billing],
    "PaymentMethod":[payment_method],
    "MonthlyCharges":[monthly_charges],
    "TotalCharges":[total_charges]
})

# ---- PREDICTION ----
if st.button("Predict Churn"):
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ This customer is likely to churn (Probability: {probability:.2f})")
    else:
        st.success(f"✅ This customer is not likely to churn (Probability: {probability:.2f})")

    # ---- FEATURE IMPORTANCE ----
    # Extract model and preprocessor from pipeline
    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocessor']

    # Get all feature names after OneHotEncoding
    num_features = preprocessor.transformers_[0][2]  # numeric columns
    cat_features = preprocessor.transformers_[1][1].get_feature_names_out(preprocessor.transformers_[1][2])
    all_features = np.concatenate([num_features, cat_features])

    # Get feature importances
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature": all_features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(5)

    st.subheader("Top 5 Features Affecting Churn")
    st.table(feat_imp_df)
