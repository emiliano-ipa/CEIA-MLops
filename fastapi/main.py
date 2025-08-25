# fastapi/main.py
import os
from typing import List, Literal, Optional, Dict, Any, Tuple
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, RootModel
import mlflow
from mlflow.tracking import MlflowClient

# =========================
# Config MLflow / MinIO
# =========================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.getenv("MODEL_EXPERIMENT", "modelos_optimizados")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

app = FastAPI(title="CEIA-MLops Model Serving", version="1.0.0")

# Simple in-memory cache
# key -> (pyfunc_model, registry_info_dict, run_info_dict)
MODEL_CACHE: Dict[str, Tuple[Any, Dict[str, Any], Dict[str, Any]]] = {}

# Aliases -> Registered Model names in MLflow
MODEL_NAME_MAP = {
    "knn": "Knn_Classifier",
    "svm": "SVC",                   # keep alias even if not present (will 404 cleanly)
    "lightgbm": "LightGBM_Classifier",
}

# ============
# Pydantic IO
# ============
class Record(RootModel):
    root: Dict[str, Any]

class PredictRequest(BaseModel):
    data: List[Record]
    columns: Optional[List[str]] = None  # optional: enforce column order/selection

# =========================
# Helpers MLflow
# =========================
def _resolve_registered_name(alias: str) -> str:
    name = MODEL_NAME_MAP.get(alias.lower())
    if not name:
        raise HTTPException(status_code=404, detail=f"Unknown model alias '{alias}'.")
    return name

def _get_latest_version_entry(reg_name: str):
    """
    MLflow stages are typically: None, 'Staging', 'Production', 'Archived'.
    You listed models with stage=None. The URI 'models:/name/latest' only works
    for a STAGE label (e.g., 'Staging'/'Production'), not "latest by number".
    So we manually pick the highest version number.
    """
    versions = client.get_latest_versions(reg_name)  # returns per-stage latest; often includes None stage
    if not versions:
        # Fallback: list all versions and pick max
        all_versions = client.search_model_versions(f"name='{reg_name}'")
        if not all_versions:
            raise HTTPException(status_code=404, detail=f"No versions found for registered model '{reg_name}'.")
        versions = list(all_versions)

    # Choose numerically largest version
    try:
        best = max(versions, key=lambda v: int(v.version))
    except Exception:
        # In case versions list has mixed types, fall back to string compare
        best = max(versions, key=lambda v: str(v.version))

    return best

def _load_model_from_registry(alias: str):
    """
    Returns: (pyfunc_model, registry_info_dict, run_info_dict)
    Caches by alias+version so we don't reload unnecessarily.
    """
    reg_name = _resolve_registered_name(alias)

    # Verify model exists
    try:
        client.get_registered_model(reg_name)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Registered model '{reg_name}' not found in MLflow Registry.")

    latest = _get_latest_version_entry(reg_name)
    version = str(latest.version)
    cache_key = f"{alias}::{version}"

    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    # Build a concrete versioned URI
    model_uri = f"models:/{reg_name}/{version}"
    print(f"[INFO] Loading model from URI: {model_uri}")

    # Load pyfunc model
    try:
        pyfunc_model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load '{model_uri}': {e}")

    # Collect registry info
    registry_info = {
        "registered_name": reg_name,
        "version": version,
        "current_stage": getattr(latest, "current_stage", None),
        "run_id": latest.run_id,
        "status": getattr(latest, "status", None),
        "source": getattr(latest, "source", None),
    }

    # Collect run metrics/params for the backing run (if available)
    run_info = {"metrics": {}, "params": {}}
    try:
        run = client.get_run(latest.run_id)
        run_info["metrics"] = run.data.metrics or {}
        run_info["params"] = run.data.params or {}
    except Exception:
        # If the run is gone or inaccessible, keep empty dicts
        pass

    MODEL_CACHE[cache_key] = (pyfunc_model, registry_info, run_info)
    return MODEL_CACHE[cache_key]

# ==========
# Endpoints
# ==========
@app.get("/health")
def health():
    return {"status": "ok", "mlflow": MLFLOW_TRACKING_URI, "experiment": EXPERIMENT_NAME}

@app.get("/list-models")
def list_models():
    """
    Lists all registered models and their latest versions (as reported by MLflow).
    """
    try:
        registered_models = client.search_registered_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing registered models: {e}")

    out = []
    for rm in registered_models:
        out.append(
            {
                "name": rm.name,
                "latest_versions": [
                    {
                        "version": v.version,
                        "stage": v.current_stage,
                        "run_id": v.run_id,
                        "status": v.status,
                    }
                    for v in rm.latest_versions
                ],
            }
        )
    return out

@app.get("/model-info")
def model_info(model: Literal["knn", "svm", "lightgbm"] = Query(...)):
    """
    Returns registry + run metadata for the latest version of the requested model alias.
    """
    try:
        _m, reg, run = _load_model_from_registry(model)
        return {
            "alias": model,
            "registered_name": reg["registered_name"],
            "version": reg["version"],
            "stage": reg["current_stage"],
            "run_id": reg["run_id"],
            "status": reg["status"],
            "source": reg["source"],
            "metrics": run["metrics"],
            "params": run["params"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def predict(
    payload: PredictRequest,
    model: Literal["knn", "svm", "lightgbm"] = Query(...),
):
    """
    Predicts using the latest registered version of the requested model alias.
    Expects JSON records; optional 'columns' enforces order/selection.
    """
    try:
        pyfunc_model, reg, _run = _load_model_from_registry(model)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    rows = [r.root for r in payload.data]
    if not rows:
        raise HTTPException(status_code=400, detail="Payload vacío.")

    df = pd.DataFrame(rows)

    if payload.columns:
        missing = [c for c in payload.columns if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Faltan columnas: {missing}")
        df = df[payload.columns]

    try:
        preds = pyfunc_model.predict(df)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error al predecir (¿features procesadas igual que en entrenamiento?): {e}",
        )

    return {
        "alias": model,
        "registered_name": reg["registered_name"],
        "version": reg["version"],
        "n_samples": len(df),
        "predictions": preds.tolist(),
    }

@app.post("/reload-cache")
def reload_cache():
    MODEL_CACHE.clear()
    return {"status": "cleared"}

@app.get("/")
def home():
    return {
        "message": "Welcome to the CEIA-MLops Model Serving API!",
        "status": "ok",
        "health_endpoint": "/health",
    }
