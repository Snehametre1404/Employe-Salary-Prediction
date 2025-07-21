import streamlit as st
import joblib

st.set_page_config(page_title="Employee Salary Prediction", layout="centered")
st.title("🧑‍💼 Employee Salary Prediction App")


try:
    model = joblib.load("knn_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
    model_loaded = True
except:
    model_loaded = False
    st.error("❌ Model or encoders not found (knn_model.pkl or label_encoders.pkl)")

if model_loaded:
    
    age = st.slider("Age", 17, 90, 30)
    workclass = st.selectbox("Workclass", encoders["workclass"].classes_)
    fnlwgt = st.number_input("Final Weight (fnlwgt)", 10000, 1000000, 400000)
    education = st.selectbox("Education", encoders["education"].classes_)
    education_num = st.slider("Educational Number", 1, 16, 10)
    marital_status = st.selectbox("Marital Status", encoders["marital-status"].classes_)
    occupation = st.selectbox("Occupation", encoders["occupation"].classes_)
    relationship = st.selectbox("Relationship", encoders["relationship"].classes_)
    race = st.selectbox("Race", encoders["race"].classes_)
    gender = st.radio("Gender", encoders["gender"].classes_)
    capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
    capital_loss = st.number_input("Capital Loss", 0, 5000, 0)
    hours_per_week = st.slider("Hours per Week", 1, 99, 40)
    native_country = st.selectbox("Native Country", encoders["native-country"].classes_)

   
    encoded_features = [
        age,
        encoders["workclass"].transform([workclass])[0],
        fnlwgt,
        encoders["education"].transform([education])[0],
        education_num,
        encoders["marital-status"].transform([marital_status])[0],
        encoders["occupation"].transform([occupation])[0],
        encoders["relationship"].transform([relationship])[0],
        encoders["race"].transform([race])[0],
        encoders["gender"].transform([gender])[0],
        capital_gain,
        capital_loss,
        hours_per_week,
        encoders["native-country"].transform([native_country])[0]
    ]

    
    if st.button("Predict Salary Class"):
        prediction = model.predict([encoded_features])[0]
        result = ">50K" if prediction == 1 else "<=50K"
        st.success(f"Predicted Salary Class: **{result}**")
