from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import joblib

# Load trained model and DictVectorizer
model = joblib.load("./heart_model.pkl")
dv = joblib.load("./Dict.pkl")

# Create FastAPI app
app = FastAPI(title="Heart Disease Prediction API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -----------------------------
# Input Data Validation Schema
# -----------------------------
class HeartInput(BaseModel):
    Age: int = Field(..., gt=0, lt=120)
    Sex: Literal["M", "F"]
    ChestPainType: Literal["ATA", "NAP", "ASY", "TA"]
    RestingBP: int = Field(..., gt=0, lt=300)
    Cholesterol: int = Field(..., gt=0, lt=600)
    FastingBS: Literal[0, 1]
    RestingECG: Literal["Normal", "ST", "LVH"]
    MaxHR: int = Field(..., gt=0, lt=250)
    ExerciseAngina: Literal["Y", "N"]
    Oldpeak: float = Field(..., ge=0, le=10)
    ST_Slope: Literal["Up", "Flat", "Down"]


# Root route
@app.get("/")
def home():
    return {"message": "Heart ML API is running"}


# -----------------------------
# Prediction route with validation
# -----------------------------
@app.post("/predict")
def predict(data: HeartInput):

    # Convert Pydantic model → dict
    input_dict = data.model_dump()

    # Transform using DictVectorizer
    X = dv.transform([input_dict])

    # Prediction
    prediction = model.predict(X)[0]

    # Probability
    probability = model.predict_proba(X)[0].max()

    return {
        "prediction": int(prediction),
        "confidence": float(probability)
    }
