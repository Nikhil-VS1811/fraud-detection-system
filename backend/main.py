from fastapi import FastAPI
from pydantic import BaseModel
from database import engine, SessionLocal,Base
from models_db import PredictionLog
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
import joblib
import shap
import numpy as np

# Load trained model
model = joblib.load("models/fraud_detection_model.pkl")
print(type(model))
explainer = shap.TreeExplainer(model)
FEATURE_NAMES = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount"
]


def get_top_contributors(features):

    shap_values = explainer.shap_values(np.array([features]))

    values = shap_values[0]

    pairs = []

    for i, value in enumerate(values):
        pairs.append({
            "feature": FEATURE_NAMES[i],
            "impact": float(abs(value)),
            "shap_value": float(value)
        })

    pairs.sort(
        key=lambda x: x["impact"],
        reverse=True
    )

    return pairs[:5]

# Create FastAPI app
Base.metadata.create_all(bind=engine)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class Transaction(BaseModel):
    features: list[float]

    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.0] * 30
            }
        }

# Health check route
@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

# Prediction route
@app.post("/predict")
def predict(transaction: Transaction):

    if len(transaction.features) != 30:
        return {"error": "Exactly 30 features required"}

    data = np.array(transaction.features).reshape(1, -1)

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    db = SessionLocal()

    log = PredictionLog(fraud_prediction=int(prediction),fraud_probability=float(probability))

    db.add(log)
    db.commit()
    db.close()
    top_features = get_top_contributors(
    transaction.features
    )

    return {
    "fraud_prediction": int(prediction),
    "fraud_probability": float(probability),
    "top_contributors": top_features
    }

@app.get("/transactions")
def get_transactions():

    db = SessionLocal()

    records = db.query(PredictionLog).all()

    db.close()

    return [
        {
            "id": record.id,
            "fraud_prediction": record.fraud_prediction,
            "fraud_probability": record.fraud_probability,
            "timestamp": record.timestamp
        }
        for record in records
    ]

@app.get("/metrics")
def get_metrics():

    return {
        "precision": 73,
        "recall": 89,
        "f1": 80,
        "roc_auc": 97.92
    }

@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }

@app.get("/model-info")
def get_model_info():
    return {
        "model": "XGBoost",
        "dataset_size": 284807,
        "features": 30,
        "fraud_samples": 492,
        "smote_applied": True,
        "roc_auc": 97.92
    }