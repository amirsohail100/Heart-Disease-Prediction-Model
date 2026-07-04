import pandas as pd
import joblib

model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("Scaler.pkl")
expected_columns = joblib.load("columns.pkl")


print("__Heart stroke prediction by Amir 💖__\n")
print("-------------👌KNN Model--------------\n")
print("---------Predict Your Result👍--------\n")
try:
    age = int(input("Age,18,100,40"))
    sex = input("SEX,[M,F]")
    chest_pain = input("Chest Pain Type,[ATA,NAP,TA,ASY]")
    resting_BP = float(input(" BP(mm Hg),80,200,120"))
    Cholesterol = float(input("Cholesterol (mg/dL),100,600,200"))
    FastingBS = int(input("Fasting BS > 120 mg/dL,[0,1]"))
    Resting_ECG = input("Resting ECG,[Normal,ST,LVH]")
    max_hr = float(input("Max HR,60,220,150"))
    ex_angina = input("Exercise-Induced Angina,[Y,N]")
    Oldpeak = float(input("Oldpeak (ST Depression),0.0,6.0,1.0"))
    ST_Slope = input("ST Slope,[Up,Flat,Down]")
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
        print("_-_-_-_-_⚠️ High Risk of Heart Disease ❤️‍🩹_-_-_-_-_\n")
    else:
        print("_-_-_-_-_✅ Low Risk of Heart Disease 💝_-_-_-_-_\n")
except Exception as e:
    print("An Error Occured", e)