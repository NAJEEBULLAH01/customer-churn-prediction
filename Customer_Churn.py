import pickle
import streamlit as st
import pandas as pd
import numpy as np


data = pickle.load(open('Customer_Churn.pkl','rb'))
model = data['model']
ct = data['ct']


st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

# Load model
@st.cache_resource
def load_model():
    data = pickle.load(open('Customer_Churn.pkl','rb'))
    return data['model'], data['ct']
model, ct = load_model()

# Header
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Enter customer details below to predict churn risk instantly.")
st.divider()

# 3 Column Layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Customer Info")
    gender = st.selectbox("Gender", ["Male","Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    Partner = st.selectbox("Partner", ["Yes","No"])
    Dependents = st.selectbox("Dependents", ["Yes","No"])

with col2:
    st.subheader("📞 Services")
    PhoneService = st.selectbox("Phone Service", ["Yes","No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes","No","No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes","No","No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes","No","No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes","No","No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes","No","No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes","No","No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes","No","No internet service"])

with col3:
    st.subheader("💳 Account & Billing")
    Contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes","No"])
    PaymentMethod = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer","Credit card"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    MonthlyCharges = st.number_input("Monthly Charges ($)", 0.0, 500.0, 70.0, step=1.0)
    TotalCharges = st.number_input("Total Charges ($)", 0.0, 100000.0, 1000.0, step=10.0)

st.divider()

# Predict
if st.button("🔍 Predict Churn Risk", use_container_width=True, type="primary"):
    input_df = pd.DataFrame([{
        'gender':gender,'Partner':Partner,'Dependents':Dependents,'PhoneService':PhoneService,
        'MultipleLines':MultipleLines,'InternetService':InternetService,'OnlineSecurity':OnlineSecurity,
        'OnlineBackup':OnlineBackup,'DeviceProtection':DeviceProtection,'TechSupport':TechSupport,
        'StreamingTV':StreamingTV,'StreamingMovies':StreamingMovies,'Contract':Contract,
        'PaperlessBilling':PaperlessBilling,'PaymentMethod':PaymentMethod,'SeniorCitizen':SeniorCitizen,
        'tenure':tenure,'MonthlyCharges':MonthlyCharges,'TotalCharges':TotalCharges
    }])

    pred = model.predict(ct.transform(input_df))[0]
    proba = model.predict_proba(ct.transform(input_df))[0][1]

    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.metric("Churn Probability", f"{proba:.2%}")
        st.progress(proba)
    with col_res2:
        if pred == 1:
            st.error("⚠️ High Risk: Customer likely to CHURN")
        else:
            st.success("✅ Low Risk: Customer likely to STAY")

st.caption("Model: Scikit-learn 1.6.1 | Built with Streamlit")