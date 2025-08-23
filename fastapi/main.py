# fastapi/main.py
import os
from typing import List, Literal, Optional, Dict, Any
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from pydantic import RootModel
import mlflow
from mlflow.tracking import MlflowClient

# =========================
# Config MLflow / MinIO
# =========================
# Fuera de Docker (localhost):
#   MLFLOW_TRACKING_URI=http://localhost:5001
# En red Docker compose:
#   MLFLOW_TRACKING_URI=http://mlflow:5000
# Artefactos en MinIO (en tu compose):
#   MLFLOW_S3_ENDPOINT_URL=http://minio:9000
#   AWS_ACCESS_KEY_ID=minio
#   AWS_SECRET_ACCESS_KEY=minio123

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.getenv("MODEL_EXPERIMENT", "modelos_optimizados")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

app = FastAPI(title="CEIA-MLops Model Serving", version="1.0.0")

# Cache simple en memoria
MODEL_CACHE: Dict[str, Any] = {}

# Mapeo nombre corto -> tag model_type que guardan tus DAGs
MODEL_TAGS = {
    "knn": "KNeighborsClassifier",
    "svm": "SVC",
    "lightgbm": "LGBMClassifier",
}


# ============
# Pydantic IO
# ============
class Record(RootModel):
    root: Dict[str, Any]


class PredictRequest(BaseModel):
    data: List[Record]
    # opcional: forzar orden de columnas
    columns: Optional[List[str]] = None


# =========================
# Helpers MLflow
# =========================
def _get_best_run_by_model(model_type_tag: str):
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not exp:
        raise RuntimeError(
            f"Experimento '{EXPERIMENT_NAME}' no existe en MLflow ({MLFLOW_TRACKING_URI})."
        )

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.model_type = '{model_type_tag}'",
        order_by=["metrics.f1_macro DESC"],
        max_results=1,
    )
    if runs.empty:
        return None
    return runs.iloc[0]


def _load_model(model_key: str):
    if model_key in MODEL_CACHE:
        return MODEL_CACHE[model_key]

    tag_value = MODEL_TAGS[model_key]
    best = _get_best_run_by_model(tag_value)
    if best is None:
        raise RuntimeError(f"No hay runs en '{EXPERIMENT_NAME}' para model_type={tag_value}")

    run_id = best.run_id
    model_uri = f"runs:/{run_id}/model"  # tus DAGs guardan el artefacto 'model'
    model = mlflow.pyfunc.load_model(model_uri)

    MODEL_CACHE[model_key] = {
        "pyfunc": model,
        "run_id": run_id,
        "metrics": {
            "f1_macro": best.get("metrics.f1_macro"),
            "precision_macro": best.get("metrics.precision_macro"),
            "recall_macro": best.get("metrics.recall_macro"),
        },
        "params": {k.replace("params.", ""): v for k, v in best.items() if k.startswith("params.")},
    }
    return MODEL_CACHE[model_key]


# ==========
# Endpoints
# ==========
@app.get("/health")
def health():
    return {"status": "ok", "mlflow": MLFLOW_TRACKING_URI, "experiment": EXPERIMENT_NAME}


@app.get("/model-info")
def model_info(model: Literal["knn", "svm", "lightgbm"] = Query(...)):
    try:
        m = _load_model(model)
        return {
            "model": model,
            "run_id": m["run_id"],
            "best_metrics": m["metrics"],
            "best_params": m["params"],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict")
def predict(
    payload: PredictRequest,
    model: Literal["knn", "svm", "lightgbm"] = Query(...),
):
    try:
        m = _load_model(model)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    rows = [r.root for r in payload.data]
    if not rows:
        raise HTTPException(status_code=400, detail="Payload vacío.")

    df = pd.DataFrame(rows)

    # si el cliente especifica columnas (orden/selección), respetarlas
    if payload.columns:
        missing = [c for c in payload.columns if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Faltan columnas: {missing}")
        df = df[payload.columns]

    try:
        preds = m["pyfunc"].predict(df)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error al predecir (¿features procesadas igual que en entrenamiento?): {e}",
        )

    return {
        "model": model,
        "run_id": m["run_id"],
        "n_samples": len(df),
        "predictions": preds.tolist(),
    }


@app.post("/reload-cache")
def reload_cache():
    MODEL_CACHE.clear()
    return {"status": "cleared"}