"""
VOC Biometric Inference Server
Run on your machine. Raspberry Pi calls this.

Usage:
    pip install fastapi uvicorn joblib scikit-learn pandas numpy
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import json
import os
import time
import logging
from datetime import datetime

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)

# ── Load model artifacts (once at startup) ───────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

log.info("Loading model artifacts...")
try:
    model         = joblib.load(f"{MODEL_DIR}/model.pkl")
    le            = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")
    top_features  = joblib.load(f"{MODEL_DIR}/top_features.pkl")

    with open(f"{MODEL_DIR}/metadata.json") as f:
        metadata = json.load(f)

    log.info(f"Model loaded | {metadata['n_persons']} persons | {metadata['n_features']} features")
    log.info(f"Persons: {metadata['persons']}")
except Exception as e:
    log.error(f"Failed to load model: {e}")
    raise RuntimeError(f"Model load failed: {e}")


# ── Request / Response schemas ───────────────────────────────
class SensorReading(BaseModel):
    """Raw stats computed on the Pi from one sampling window."""
    # MQ6 sensor fields
    mq6_1_min    : float
    mq6_1_mean   : float
    mq6_1_max    : float
    mq6_1_std    : float
    mq6_1_median : float
    mq6_1_iqr    : float
    mq6_1_skew   : float

    # MEMS odor sensor fields
    mems_odor_1_min      : float
    mems_odor_1_mean     : float
    mems_odor_1_max      : float
    mems_odor_1_std      : float
    mems_odor_1_median   : float
    mems_odor_1_iqr      : float
    mems_odor_1_skew     : float
    mems_odor_1_kurtosis : float
    mems_odor_1_cv       : float
    mems_odor_1_energy   : float

    # Optional context
    round_no    : Optional[int]   = None
    device_id   : Optional[str]   = None   # which Pi sent this
    timestamp   : Optional[str]   = None


class PredictionResponse(BaseModel):
    person        : str
    confidence    : float
    status        : str          # "identified" | "uncertain" | "unknown"
    all_probs     : dict
    latency_ms    : float
    timestamp     : str


# ── Feature engineering (mirrors Step 4 from notebook) ───────
def engineer_features(row: dict) -> pd.DataFrame:
    df = pd.DataFrame([row])

    df['mq6_mems_ratio']     = df['mq6_1_mean']      / (df['mems_odor_1_mean'] + 1e-9)
    df['mq6_mems_std_ratio'] = df['mq6_1_std']       / (df['mems_odor_1_std']  + 1e-9)
    df['mq6_cv']             = df['mq6_1_std']       / (df['mq6_1_mean']        + 1e-9)
    df['range_mq6']          = df['mq6_1_max']       - df['mq6_1_min']
    df['range_mems']         = df['mems_odor_1_max'] - df['mems_odor_1_min']
    df['skew_diff']          = df['mq6_1_skew']      - df['mems_odor_1_skew']
    df['mq6_delta']          = 0   # no previous round context
    df['mems_delta']         = 0
    df['mq6_delta2']         = 0
    df['mems_delta2']        = 0

    # Fill any features the model expects but aren't computed
    for feat in top_features:
        if feat not in df.columns:
            df[feat] = 0

    return df[top_features].fillna(0)


# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="VOC Biometric API",
    description="Identify a person from VOC sensor readings",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIDENCE_THRESHOLD = 0.50   # below this → "uncertain"


# ── Routes ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service"   : "VOC Biometric API",
        "status"    : "running",
        "persons"   : metadata["persons"],
        "n_features": metadata["n_features"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": type(model).__name__, "timestamp": datetime.now().isoformat()}


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    t_start = time.perf_counter()

    try:
        # Engineer features
        X = engineer_features(reading.dict())

        # Inference
        X_np = X.values   # numpy → avoids feature_names_in_ issues
        if hasattr(model, "predict_proba"):
            probs      = model.predict_proba(X_np)[0]
            pred_idx   = int(probs.argmax())
            confidence = float(probs[pred_idx])
            all_probs  = {str(cls): round(float(p), 4)
                          for cls, p in zip(le.classes_, probs)}
        else:
            pred_idx   = int(model.predict(X_np)[0])
            confidence = 1.0
            all_probs  = {}

        person = str(le.inverse_transform([pred_idx])[0])
        status = "identified" if confidence >= CONFIDENCE_THRESHOLD else "uncertain"

    except Exception as e:
        log.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.perf_counter() - t_start) * 1000

    log.info(
        f"[{reading.device_id or 'unknown'}] "
        f"→ {person} ({confidence:.3f}) | {latency_ms:.1f}ms | {status}"
    )

    return PredictionResponse(
        person     = person,
        confidence = round(confidence, 4),
        status     = status,
        all_probs  = all_probs,
        latency_ms = round(latency_ms, 2),
        timestamp  = datetime.now().isoformat(),
    )


@app.get("/persons")
def list_persons():
    """Returns list of all known persons the model can identify."""
    return {"persons": metadata["persons"]}
