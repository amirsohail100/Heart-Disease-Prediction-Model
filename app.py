import pandas as pd
import joblib
import streamlit as st

model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("Scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("Heart stroke prediction by Amir ❤️")
st.markdown("Provide the following details")

age = st.slider("Age",18,100,40)
sex = st.selectbox("SEX",["M","F"])
chest_pain = st.selectbox("Chest Pain Type",["ATA","NAP","TA","ASY"])
resting_BP = st.number_input(" BP(mm Hg)",80,200,120)
Cholesterol = st.number_input("Cholesterol (mg/dL)",100,600,200)
FastingBS = st.selectbox("Fasting BS > 120 mg/dL",[0,1])
Resting_ECG = st.selectbox("Resting ECG",["Normal","ST","LVH"])
max_hr = st.slider("Max HR",60,220,150)
ex_angina = st.selectbox("Exercise-Induced Angina",["Y","N"])
Oldpeak = st.slider("Oldpeak (ST Depression)",0.0,6.0,1.0)
ST_Slope = st.selectbox("ST Slope",["Up","Flat","Down"])
if st.button("Predict"):
    raw_input = {
        "Age":age,
        "RestingBP":resting_BP,
        "Cholesterol":Cholesterol,
        "MaxHR":max_hr,
        "Oldpeak":Oldpeak,
        "Sex_" + sex:1,
        "ChestPainType_" +chest_pain:1,
        "RestingECG_" +Resting_ECG:1,
        "ExerciseAngina_" +ex_angina:1,
        "ST_Slope_" +ST_Slope:1
    }
    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]
    Scaler_input = scaler.transform(input_df)
    prediction = model.predict(Scaler_input)[0]
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

