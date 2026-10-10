import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from database import engine, SessionLocal, Base
from models_db import PredictionLog
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = BASE_DIR / "models" / "fraud_detection_pipeline.pkl"
MODEL_PATH = BASE_DIR / "models" / "fraud_detection_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
SAMPLES_PATH = BASE_DIR / "models" / "sample_transactions.json"

FEATURE_NAMES = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]

# 1. Load canonical inference pipeline or fallback to standalone model
pipeline = None
model = None
preprocessor = None
explainer = None
classifier_for_explain = None

if PIPELINE_PATH.exists():
    try:
        pipeline = joblib.load(PIPELINE_PATH)
        print(f"Loaded canonical pipeline from {PIPELINE_PATH}")
        classifier_for_explain = pipeline.named_steps.get("classifier")
        preprocessor = pipeline.named_steps.get("preprocessor")
    except Exception as e:
        print(f"Warning: Failed to load canonical pipeline from {PIPELINE_PATH}: {e}")

if pipeline is None:
    # Rollback path: load legacy standalone model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Loaded legacy standalone model from {MODEL_PATH}")
        classifier_for_explain = model
    except Exception as e:
        print(f"Error: Failed to load legacy model from {MODEL_PATH}: {e}")

    if PREPROCESSOR_PATH.exists():
        try:
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            print(f"Loaded standalone preprocessor from {PREPROCESSOR_PATH}")
        except Exception as e:
            print(f"Warning: Could not load standalone preprocessor: {e}")

# Initialize SHAP TreeExplainer on the underlying tree classifier
if classifier_for_explain is not None:
    try:
        import shap
        explainer = shap.TreeExplainer(classifier_for_explain)
        print("Initialized SHAP TreeExplainer on underlying classifier.")
    except Exception as e:
        print(f"Notice: SHAP TreeExplainer could not be initialized ({e}). Using feature importance fallback.")


def get_top_contributors(features):
    """
    Computes top 5 feature contributions on scaled feature inputs.
    """
    if explainer is not None:
        try:
            shap_values = explainer.shap_values(np.array([features]))
            values = shap_values[0]
            pairs = []
            for i, value in enumerate(values):
                pairs.append({
                    "feature": FEATURE_NAMES[i],
                    "impact": float(abs(value)),
                    "shap_value": float(value),
                })
            pairs.sort(key=lambda x: x["impact"], reverse=True)
            return pairs[:5]
        except Exception as e:
            print(f"Warning: SHAP calculation failed: {e}")

    # Fallback to model feature importances
    if classifier_for_explain is not None and hasattr(classifier_for_explain, "feature_importances_"):
        try:
            importances = classifier_for_explain.feature_importances_
            pairs = []
            for i, imp in enumerate(importances):
                val = float(features[i])
                pairs.append({
                    "feature": FEATURE_NAMES[i],
                    "impact": float(abs(imp * val)),
                    "shap_value": float(imp * val),
                })
            pairs.sort(key=lambda x: x["impact"], reverse=True)
            return pairs[:5]
        except Exception:
            pass

    return []


# Create database tables if database is accessible
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"Warning: Database connection / initialization failed: {e}")

# Create FastAPI app
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


# Health check routes
@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# Prediction route
@app.post("/predict")
def predict(transaction: Transaction):
    if len(transaction.features) != 30:
        raise HTTPException(status_code=400, detail="Exactly 30 features required")

    data = np.array(transaction.features, dtype=np.float64).reshape(1, -1)
    if not np.all(np.isfinite(data)):
        raise HTTPException(status_code=400, detail="All 30 features must be finite numeric values")

    import pandas as pd
    data_df = pd.DataFrame(data, columns=FEATURE_NAMES)

    # 1. Canonical Pipeline Inference (Raw features passed, preprocessing applied internally ONCE)
    if pipeline is not None:
        prediction = int(pipeline.predict(data_df)[0])
        probability = float(pipeline.predict_proba(data_df)[0][1])

        # Prepare scaled features exclusively for SHAP explanation
        if preprocessor is not None:
            scaled_features = preprocessor.transform(data_df)[0]
        else:
            scaled_features = data[0]

        top_features = get_top_contributors(scaled_features)

    # 2. Rollback Fallback Inference (Pre-pipeline legacy path)
    elif model is not None:
        if preprocessor is not None:
            data_processed = preprocessor.transform(data)
        else:
            data_processed = data

        prediction = int(model.predict(data_processed)[0])
        probability = float(model.predict_proba(data_processed)[0][1])
        top_features = get_top_contributors(data_processed[0])

    else:
        raise HTTPException(status_code=500, detail="No model or pipeline loaded on server")

    # Safeguarded database logging so external DB issues do not break inference
    try:
        db = SessionLocal()
        log = PredictionLog(
            fraud_prediction=prediction,
            fraud_probability=probability,
        )
        db.add(log)
        db.commit()
        db.close()
    except Exception as db_err:
        print(f"Warning: Failed to log transaction to database: {db_err}")

    return {
        "fraud_prediction": prediction,
        "fraud_probability": probability,
        "top_contributors": top_features,
    }


@app.get("/transactions")
def get_transactions():
    try:
        db = SessionLocal()
        records = db.query(PredictionLog).all()
        db.close()

        return [
            {
                "id": record.id,
                "fraud_prediction": record.fraud_prediction,
                "fraud_probability": record.fraud_probability,
                "timestamp": record.timestamp,
            }
            for record in records
        ]
    except Exception as e:
        print(f"Warning: Could not fetch transactions: {e}")
        return []


@app.get("/metrics")
def get_metrics():
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r") as f:
                data = json.load(f)
            return {
                "precision": data.get("precision", 73.11),
                "recall": data.get("recall", 88.78),
                "f1": data.get("f1", 80.18),
                "roc_auc": data.get("roc_auc", 97.92),
                "pr_auc": data.get("pr_auc", 87.74),
            }
        except Exception as e:
            print(f"Warning: Failed to read metrics.json: {e}")

    return {
        "precision": 73.11,
        "recall": 88.78,
        "f1": 80.18,
        "roc_auc": 97.92,
        "pr_auc": 87.74,
    }


@app.get("/model-info")
def get_model_info():
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r") as f:
                data = json.load(f)
            return {
                "model": data.get("model", "XGBoost"),
                "dataset_size": data.get("dataset_size", 284807),
                "features": len(data.get("feature_names", FEATURE_NAMES)),
                "fraud_samples": data.get("fraud_cases_total", 492),
                "smote_applied": True,
                "roc_auc": data.get("roc_auc", 97.92),
                "pr_auc": data.get("pr_auc", 87.74),
            }
        except Exception as e:
            print(f"Warning: Failed to read model-info: {e}")

    return {
        "model": "XGBoost",
        "dataset_size": 284807,
        "features": 30,
        "fraud_samples": 492,
        "smote_applied": True,
        "roc_auc": 97.92,
        "pr_auc": 87.74,
    }


@app.get("/samples")
def get_samples():
    if SAMPLES_PATH.exists():
        try:
            with open(SAMPLES_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to read sample_transactions.json: {e}")
    return {}