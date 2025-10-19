import os
import shutil
import tempfile
import threading
import logging
import pandas as pd
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Body

from . import model_manager, utils
from .schemas import PredictRequest, PredictResponse, RetrainJsonRequest

app = FastAPI(title="TruValue API", version="1.0")
logger = logging.getLogger("truvalue_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

retrain_lock = threading.Lock()
start_time = datetime.utcnow()
metrics = {
    "prediction_count": 0,
    "last_retrain_time": None,
    "last_model_version": None
}

@app.get("/")
def root():
    return {"message": "Welcome to the TruValue API. Use /docs for usage."}

@app.on_event("startup")
def startup_event():
    try:
        model_manager.load_model()
        metrics["last_model_version"] = model_manager._model_version
        logger.info(f"Model version {model_manager._model_version} loaded successfully at startup.")
    except Exception as e:
        logger.error(f"Startup warning: {e}")

@app.get("/health")
def health():
    uptime = (datetime.utcnow() - start_time).total_seconds()
    version = getattr(model_manager, "_model_version", "not_loaded")
    return {
        "status": "ok",
        "model_loaded": model_manager._model is not None,
        "model_version": version,
        "uptime_seconds": int(uptime)
    }

@app.get("/metrics")
def get_metrics():
    return {
        "prediction_count": metrics["prediction_count"],
        "last_retrain_time": metrics["last_retrain_time"],
        "current_model_version": metrics["last_model_version"]
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        if model_manager._model is None:
            raise FileNotFoundError("Model not loaded.")
        if not hasattr(model_manager._model, "predict"):
            raise ValueError("Loaded object is not a valid model.")

        input_dict = req.dict()
        df = pd.DataFrame([input_dict])
        logger.info(f"Received prediction request: {input_dict}")

        # Use preprocessor if present
        df_proc = model_manager._preprocessor.transform(df) if hasattr(model_manager, "_preprocessor") else df
        preds = model_manager._model.predict(df_proc)
        pred = float(preds[0])

        # Post-process & sanitize
        pred = utils.sanitize_prediction(pred)
        pred = max(pred, 0.0)  # Business logic: Ensure non-negative

        version = model_manager._model_version or "unknown"
        metrics["prediction_count"] += 1
        metrics["last_model_version"] = version

        logger.info(f"Prediction={pred} (model={version})")
        return PredictResponse(prediction=pred, model_version=version)
    except FileNotFoundError as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

def retrain_background(df: pd.DataFrame, version_name: Optional[str] = None):
    try:
        logger.info("Retraining started in background.")
        result = model_manager.retrain_from_dataframe(df, version_name)
        with retrain_lock:
            metrics["last_retrain_time"] = datetime.utcnow().isoformat()
            metrics["last_model_version"] = str(result)
        logger.info(f"Retraining complete. New model: {result}")
        return result
    except Exception as e:
        logger.exception(f"Retraining failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {e}")


# CSV retrain endpoint
@app.post("/retrain_csv")
async def retrain_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Retrain with CSV file upload (form-data).
    Upload a CSV file matching the training schema.
    """
    with retrain_lock:
        try:
            suffix = os.path.splitext(file.filename)[1] if file.filename else ".csv"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_path = tmp.name
                logger.info(f"Uploaded retrain file saved to {tmp_path}")
            df = pd.read_csv(tmp_path)
            background_tasks.add_task(retrain_background, df, None)
            os.remove(tmp_path)
            return {"status": "retraining_started_from_csv", "message": "Running in background."}
        except Exception as e:
            logger.exception(f"Failed to retrain from uploaded file: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to retrain: {e}")

# JSON retrain endpoint
@app.post("/retrain_json")
async def retrain_json(
    background_tasks: BackgroundTasks,
    json_payload: RetrainJsonRequest = Body(...)
):
    """
    Retrain with application/json payload.
    Pass a JSON like: {"data": [...], "version_name": "..."}
    """
    with retrain_lock:
        try:
            df = pd.DataFrame(json_payload.data)
            background_tasks.add_task(retrain_background, df, json_payload.version_name)
            return {"status": "retraining_started_from_json", "message": "Running in background."}
        except Exception as e:
            logger.exception(f"Failed to retrain from JSON: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to retrain from JSON: {e}")
