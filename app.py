"""
FastAPI Backend for Fraud Detection System
Provides REST API endpoints for real-time fraud analysis, model metrics, and transaction history.
"""

import json
import os
import time
import random
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from fraud_detection.data_generator import generate_transaction_dataset, FEATURE_COLUMNS
from fraud_detection.models import (
    IsolationForestDetector, RandomForestDetector,
    XGBoostDetector, AutoencoderDetector, train_all_models
)

# ─── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Fraud Detection API",
    description="Real-time financial transaction fraud detection powered by ML & Deep Learning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# ─── Global State ─────────────────────────────────────────────────────────────
MODEL_DIR = Path("models")
models = {}
training_results = {}
transaction_log: List[dict] = []   # In-memory transaction history
TRANSACTION_LIMIT = 1000           # Max log size

MERCHANT_MAP = {
    0: "Grocery", 1: "E-Commerce", 2: "Restaurant",
    3: "Fuel Station", 4: "Electronics", 5: "Travel/Hotel"
}


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────
class TransactionInput(BaseModel):
    amount:                  float = Field(..., gt=0, description="Transaction amount in USD")
    hour_of_day:             int   = Field(..., ge=0, le=23)
    day_of_week:             int   = Field(..., ge=0, le=6)
    merchant_category:       int   = Field(..., ge=0, le=5)
    transaction_frequency:   int   = Field(..., ge=0)
    avg_transaction_amount:  float = Field(..., gt=0)
    distance_from_home_km:   float = Field(..., ge=0)
    is_foreign_transaction:  int   = Field(..., ge=0, le=1)
    num_transactions_1h:     int   = Field(..., ge=0)
    num_transactions_24h:    int   = Field(..., ge=0)
    account_age_days:        int   = Field(..., ge=0)
    credit_limit:            float = Field(..., gt=0)
    card_present:            int   = Field(..., ge=0, le=1)
    pin_used:                int   = Field(..., ge=0, le=1)
    online_transaction:      int   = Field(..., ge=0, le=1)


def _build_feature_vector(t: TransactionInput) -> np.ndarray:
    """Convert pydantic schema → numpy feature array."""
    base = {k: getattr(t, k) for k in FEATURE_COLUMNS[:15]}
    base['amount_to_avg_ratio']  = t.amount / (t.avg_transaction_amount + 1e-9)
    base['credit_utilization']   = t.amount / (t.credit_limit + 1)
    base['is_night_transaction'] = int(t.hour_of_day >= 22 or t.hour_of_day <= 5)
    base['high_velocity']        = int(t.num_transactions_1h > 3)
    base['is_weekend']           = int(t.day_of_week >= 5)
    return np.array([[base[f] for f in FEATURE_COLUMNS]])


def _load_models():
    """Attempt to load all persisted models."""
    global models, training_results
    try:
        models['isolation_forest'] = IsolationForestDetector.load()
    except Exception:
        pass
    try:
        models['random_forest'] = RandomForestDetector.load()
    except Exception:
        pass
    try:
        models['xgboost'] = XGBoostDetector.load()
    except Exception:
        pass
    try:
        models['autoencoder'] = AutoencoderDetector.load()
    except Exception:
        pass
    results_path = MODEL_DIR / "training_results.json"
    if results_path.exists():
        with open(results_path) as f:
            training_results = json.load(f)


@app.on_event("startup")
async def startup_event():
    _load_models()
    print(f"✅ Loaded {len(models)} model(s): {list(models.keys())}")


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("frontend/index.html")


@app.get("/api/health")
async def health():
    return {
        "status": "online",
        "models_loaded": list(models.keys()),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/api/train")
async def train_models_endpoint(background_tasks: BackgroundTasks):
    """Kick off model training in the background."""
    def _train():
        global models, training_results
        result = train_all_models(n_samples=15000)
        training_results = result
        _load_models()
    background_tasks.add_task(_train)
    return {"message": "Training started in background. Check /api/metrics when complete."}


@app.post("/api/predict")
async def predict(transaction: TransactionInput):
    """Analyse a single transaction and return fraud probability from all models."""
    if not models:
        raise HTTPException(status_code=503, detail="No models loaded. Please train first via POST /api/train")

    X = _build_feature_vector(transaction)
    result = {"timestamp": datetime.utcnow().isoformat(), "predictions": {}, "ensemble_score": 0.0}
    weights = {"isolation_forest": 0.15, "random_forest": 0.30, "xgboost": 0.40, "autoencoder": 0.15}
    total_weight = 0.0
    ensemble_score = 0.0

    for name, model in models.items():
        try:
            if name == 'isolation_forest':
                prob = float(model.score_samples(X)[0])
                pred = int(model.predict(X)[0])
            else:
                prob = float(model.predict_proba(X)[0])
                pred = int(prob >= 0.5)
            result['predictions'][name] = {"probability": round(prob, 4), "is_fraud": bool(pred)}
            ensemble_score += prob * weights.get(name, 0.25)
            total_weight   += weights.get(name, 0.25)
        except Exception as e:
            result['predictions'][name] = {"error": str(e)}

    if total_weight > 0:
        ensemble_score /= total_weight  # Normalize in case some models missing

    result['ensemble_score']  = round(ensemble_score, 4)
    result['is_fraud']        = bool(ensemble_score >= 0.5)
    result['risk_level']      = (
        "CRITICAL" if ensemble_score >= 0.75 else
        "HIGH"     if ensemble_score >= 0.5  else
        "MEDIUM"   if ensemble_score >= 0.25 else
        "LOW"
    )
    result['merchant'] = MERCHANT_MAP.get(transaction.merchant_category, "Unknown")

    # Log transaction
    log_entry = {**transaction.dict(), **result, "id": f"TXN{random.randint(10000000, 99999999)}"}
    transaction_log.append(log_entry)
    if len(transaction_log) > TRANSACTION_LIMIT:
        transaction_log.pop(0)

    return result


@app.get("/api/predict/random")
async def predict_random():
    """Generate and analyse a random transaction (for demo purposes)."""
    df = generate_transaction_dataset(n_samples=1, fraud_ratio=random.uniform(0.0, 1.0))
    row = df.iloc[0]
    txn = TransactionInput(**{k: float(row[k]) for k in FEATURE_COLUMNS[:15]})
    result = await predict(txn)
    result['ground_truth'] = bool(row['is_fraud'])
    result['transaction_details'] = {k: float(row[k]) for k in FEATURE_COLUMNS}
    return result


@app.get("/api/metrics")
async def get_metrics():
    """Return training metrics for all models."""
    if not training_results:
        raise HTTPException(status_code=404, detail="No training results found. Train models first.")
    return training_results


@app.get("/api/transactions")
async def get_transactions(limit: int = 50):
    """Return recent transaction history."""
    return {"transactions": transaction_log[-limit:], "total": len(transaction_log)}


@app.get("/api/simulate")
async def simulate_transactions(n: int = 20):
    """
    Simulate a batch of transactions and return analysis.
    Used to populate the dashboard with realistic data.
    """
    df = generate_transaction_dataset(n_samples=n, fraud_ratio=0.20)
    results = []
    for _, row in df.iterrows():
        try:
            txn = TransactionInput(**{k: float(row[k]) for k in FEATURE_COLUMNS[:15]})
            res = await predict(txn)
            res['amount']      = float(row['amount'])
            res['merchant']    = MERCHANT_MAP.get(int(row['merchant_category']), "Unknown")
            res['ground_truth'] = bool(row['is_fraud'])
            results.append(res)
        except Exception:
            pass
    fraud_count = sum(1 for r in results if r.get('is_fraud'))
    return {
        "total": len(results),
        "fraud_detected": fraud_count,
        "legit_count": len(results) - fraud_count,
        "transactions": results
    }


@app.get("/api/feature_importance")
async def feature_importance():
    """Return feature importance from tree-based models."""
    out = {}
    for name in ['random_forest', 'xgboost']:
        model = models.get(name)
        if model and hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
            fi = model.feature_importances_.tolist()
            out[name] = [{"feature": f, "importance": round(v, 6)}
                         for f, v in sorted(zip(FEATURE_COLUMNS, fi), key=lambda x: -x[1])]
    return out


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
