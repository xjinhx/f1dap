from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

app = FastAPI(title="F1 Qualifying Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "https://f1dap.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and feature list on startup
MODEL_PATH = Path(__file__).parent / "model.pkl"
FEATURES_PATH = Path(__file__).parent / "features.pkl"

model = None
feature_names = None

@app.on_event("startup")
def load_model():
    global model, feature_names
    if not MODEL_PATH.exists():
        print("WARNING: model.pkl not found. Place it in the backend/ folder.")
        return
    model = joblib.load(MODEL_PATH)
    if FEATURES_PATH.exists():
        feature_names = joblib.load(FEATURES_PATH)
    else:
        # fallback to known feature order from notebook
        feature_names = [
            "n_fp_laps",
            "n_fp_sessions_participated",
            "best_fp_lap_time_overall",
            "avg_fp_lap_time",
            "median_fp_lap_time",
            "best_last_fp_lap_time",
            "best_last_fp_s1",
            "best_last_fp_s2",
            "best_last_fp_s3",
        ]
    print(f"Model loaded. Features: {feature_names}")


class PredictRequest(BaseModel):
    n_fp_laps: float
    n_fp_sessions_participated: float
    best_fp_lap_time_overall: float
    avg_fp_lap_time: float
    median_fp_lap_time: float
    best_last_fp_lap_time: float
    best_last_fp_s1: float
    best_last_fp_s2: float
    best_last_fp_s3: float


class PredictResponse(BaseModel):
    predicted_position: float
    rounded_position: int


@app.get("/")
def root():
    return {"status": "F1 Qualifying Predictor API is running"}


@app.get("/health")
def health():
    return {"model_loaded": model is not None, "features": feature_names}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Add model.pkl to backend/")

    input_data = pd.DataFrame([{
        "n_fp_laps": req.n_fp_laps,
        "n_fp_sessions_participated": req.n_fp_sessions_participated,
        "best_fp_lap_time_overall": req.best_fp_lap_time_overall,
        "avg_fp_lap_time": req.avg_fp_lap_time,
        "median_fp_lap_time": req.median_fp_lap_time,
        "best_last_fp_lap_time": req.best_last_fp_lap_time,
        "best_last_fp_s1": req.best_last_fp_s1,
        "best_last_fp_s2": req.best_last_fp_s2,
        "best_last_fp_s3": req.best_last_fp_s3,
    }])

    # Ensure column order matches training
    if feature_names:
        input_data = input_data[feature_names]

    prediction = model.predict(input_data)[0]
    rounded = max(1, min(20, round(float(prediction))))

    return PredictResponse(
        predicted_position=round(float(prediction), 2),
        rounded_position=rounded,
    )
