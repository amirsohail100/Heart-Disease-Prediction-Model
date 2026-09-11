from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# Model load kar rahe hain
model = load("model_pipeline.pkl")

app = FastAPI(title="Heart Disease Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Heart Failure Dataset ke real columns aur values ke hisaab se Pydantic Model
class Data(BaseModel):
    Age: int = Field(..., ge=1, le=120, description="Age of the patient")
    Sex: Literal['M', 'F'] = Field(..., description="Sex of the patient")
    ChestPainType: Literal['ATA', 'NAP', 'ASY', 'TA'] = Field(..., description="Chest pain type")
    RestingBP: float = Field(..., ge=0, le=300, description="Resting blood pressure (mm Hg)")
    Cholesterol: float = Field(..., ge=0, le=600, description="Serum cholesterol in mg/dl")
    FastingBS: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)")
    RestingECG: Literal['Normal', 'ST', 'LVH'] = Field(..., description="Resting electrocardiographic results")
    MaxHR: float = Field(..., ge=0, le=250, description="Maximum heart rate achieved")
    ExerciseAngina: Literal['N', 'Y'] = Field(..., description="Exercise-induced angina")
    Oldpeak: float = Field(..., ge=-5.0, le=10.0, description="Oldpeak = ST depression induced by exercise relative to rest")
    ST_Slope: Literal['Up', 'Flat', 'Down'] = Field(..., description="The slope of the peak exercise ST segment")

# Professional Response Structure
class PredictionResponse(BaseModel):
    prediction_label: Literal['High Risk', 'Low Risk']
    heart_disease_probability: float = Field(..., description="Confidence score of the prediction")

@app.get("/")
def greet():
    return {"message": "Welcome to Amir WED - Heart Disease Prediction API"}

@app.post('/predict', response_model=PredictionResponse)
def predict(data: Data):
    # Input data ko DataFrame mein convert karna (Exact dataset column names ke sath)
    input_row = pd.DataFrame([{
        "Age": data.Age,
        "Sex": data.Sex,
        "ChestPainType": data.ChestPainType,
        "RestingBP": data.RestingBP,
        "Cholesterol": data.Cholesterol,
        "FastingBS": data.FastingBS,
        "RestingECG": data.RestingECG,
        "MaxHR": data.MaxHR,
        "ExerciseAngina": data.ExerciseAngina,
        "Oldpeak": data.Oldpeak,
        "ST_Slope": data.ST_Slope
    }])
    
    # Model prediction (0 or 1)
    prediction = model.predict(input_row)[0]
    
    # Probability nikalna agar model support kare
    try:
        probabilities = model.predict_proba(input_row)[0]
        confidence = float(probabilities[1] if prediction == 1 else probabilities[0])
    except Exception:
        confidence = 1.0 if prediction == 1 else 0.0

    # Professional Risk Labels
    if prediction == 1:
        label = "High Risk"
    else:
        label = "Low Risk"
        
    return PredictionResponse(
        prediction_label=label,
        heart_disease_probability=round(confidence, 4)
    )