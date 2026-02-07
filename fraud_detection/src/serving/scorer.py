import os
import json
import joblib
import numpy as np
import mlflow
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------
# App
# ---------------------------------------------------------
app = FastAPI(title="Fraud Detection Scorer")


os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "fraud_detection_experiment"
RUN_NAME = "xgboost_baseline"

ARTIFACT_DIR = "C:/Fraud-Dectection-Pipeline/fraud_detection/models/"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ---------------------------------------------------------
# Globals (loaded at startup)
# ---------------------------------------------------------
model = None
preprocessor = None
threshold = None

# ---------------------------------------------------------
# MLflow loader
# ---------------------------------------------------------
def load_latest_artifacts():
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)

    if exp is None:
        raise RuntimeError("MLflow experiment not found")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.mlflow.runName = '{RUN_NAME}'",
        order_by=["start_time DESC"],
        max_results=1
    )

    if not runs:
        raise RuntimeError("No MLflow run found")

    run = runs[0]

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    model_path = client.download_artifacts(
        run.info.run_id,
        "model/xgb_model.joblib",
        ARTIFACT_DIR
    )

    preprocess_meta_path = client.download_artifacts(
        run.info.run_id,
        "preprocess/preprocess_meta.joblib",
        ARTIFACT_DIR
    )

    model = joblib.load(model_path)
    preprocess_meta = joblib.load(preprocess_meta_path)

    threshold = run.data.metrics.get("best_threshold", 0.5)

    return model, preprocess_meta, threshold

# ---------------------------------------------------------
# Startup hook (CRITICAL for Windows)
# ---------------------------------------------------------
@app.on_event("startup")
def startup_event():
    global model, preprocessor, threshold

    model, preprocess_meta, threshold = load_latest_artifacts()
    preprocessor = preprocess_meta["preprocessor"]

    print("Model and preprocessor loaded successfully")

# ---------------------------------------------------------
# Request schema
# ---------------------------------------------------------
class TxRequest(BaseModel):
    tx_id: str
    user_id: int
    amount: float
    device: str
    ip_hash: int
    metadata: dict = {}
    label_ts:str

# ---------------------------------------------------------
# Feature builder
# ---------------------------------------------------------
def build_feature_row(tx: TxRequest):
    return {
        "amount": tx.amount,
        "device": tx.device,
        "ip_hash": tx.ip_hash,
        "metadata": json.dumps(tx.metadata),
        'label_ts': tx.label_ts
    }

# ---------------------------------------------------------
# Health
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------------------------------------------------
# Scoring endpoint
# ---------------------------------------------------------
@app.post("/score")
def score(tx: TxRequest):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        row = build_feature_row(tx)
        df = pd.DataFrame([row])

        x = preprocessor.transform(df)
        x = np.asarray(x)

        proba = float(model.predict_proba(x)[:, 1][0])
        decision = int(proba >= threshold)

        return {
            "tx_id": tx.tx_id,
            "score": proba,
            "decision": decision,
            "threshold": threshold
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
