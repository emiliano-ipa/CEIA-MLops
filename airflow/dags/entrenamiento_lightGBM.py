"""
DAG: train_lightgbm
-------------------

Este DAG entrena un modelo **LightGBM** para clasificación multiclase.
Se optimizan hiperparámetros con Optuna y se registran los resultados en MLflow.

Flujo principal:
1. Obtención de datasets desde MinIO.
2. Optimización de hiperparámetros clave de LightGBM (num_leaves, learning_rate, n_estimators).
3. Entrenamiento del modelo final.
4. Log de métricas (F1, Recall, Precision) y matriz de confusión.
5. Registro del modelo final en MLflow como artefacto versionado.

Tags: ml, optuna, minio, multiclase
"""

from airflow.decorators import dag, task
from datetime import datetime
from minio import Minio
from io import BytesIO
import pandas as pd
import mlflow
import mlflow.sklearn
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import optuna
import os
import logging


LOGGER = logging.getLogger(__name__)


def _minio_client() -> Minio:
    return Minio("minio:9000", access_key="minio", secret_key="minio123", secure=False)


def _read_csv_from_minio(bucket: str, key: str) -> pd.DataFrame:
    obj = _minio_client().get_object(bucket, key)
    return pd.read_csv(BytesIO(obj.read()))


@dag(
    dag_id="train_lightgbm",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    description="Entrena modelo LightGBM con Optuna y registra el mejor en MLflow",
    tags=["ml", "optuna", "minio", "multiclase", "lightgbm"],
)
def lightgbm_direct_dag():
    @task(task_id="load_data_meta")
    def load_data_meta() -> dict:
        """Devuelve referencias a los datos procesados en MinIO (sin empujar datasets por XCom)."""
        bucket = "processed"
        keys = {
            "X_train": "X_train.csv",
            "y_train": "y_train.csv",
            "X_test": "X_test.csv",
            "y_test": "y_test.csv",
        }
        client = _minio_client()
        for key in keys.values():
            client.stat_object(bucket, key)
        LOGGER.info("Datos disponibles en MinIO (bucket=%s): %s", bucket, keys)
        return {"bucket": bucket, "keys": keys}

    @task(task_id="train_lightgbm")
    def train_lightgbm(meta: dict) -> None:
        """Lee datos, corre Optuna (10 trials) y registra el mejor LGBMClassifier."""
        bucket, keys = meta["bucket"], meta["keys"]
        X_train = _read_csv_from_minio(bucket, keys["X_train"])
        y_train = _read_csv_from_minio(bucket, keys["y_train"]).squeeze()
        X_test = _read_csv_from_minio(bucket, keys["X_test"])
        y_test = _read_csv_from_minio(bucket, keys["y_test"]).squeeze()

        LOGGER.info(
            "Shapes: X_train=%s, X_test=%s, y_train=%s, y_test=%s",
            X_train.shape,
            X_test.shape,
            y_train.shape,
            y_test.shape,
        )

        mlflow_port = os.getenv("MLFLOW_PORT", "5000")
        mlflow.set_tracking_uri(f"http://mlflow:{mlflow_port}")

        # Optuna: experimento específico
        mlflow.set_experiment("lightgbm_optuna")
        optuna.logging.set_verbosity(optuna.logging.INFO)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 400),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", -1, 16),
                "num_leaves": trial.suggest_int("num_leaves", 8, 128),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "random_state": 42,
            }
            model = LGBMClassifier(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            f1 = f1_score(y_test, preds, average="macro")

            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("f1_macro", f1)

            LOGGER.info("Trial %s -> f1_macro=%.4f, params=%s", trial.number, f1, params)
            return f1

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)

        best_params = study.best_params
        final_model = LGBMClassifier(**best_params)
        final_model.fit(X_train, y_train)
        preds = final_model.predict(X_test)

        f1 = f1_score(y_test, preds, average="macro")
        recall = recall_score(y_test, preds, average="macro")
        precision = precision_score(y_test, preds, average="macro")

        mlflow.set_experiment("modelos_optimizados")
        with mlflow.start_run():
            mlflow.set_tag("model_type", "LGBMClassifier")
            mlflow.log_params(best_params)
            mlflow.log_metric("f1_macro", f1)
            mlflow.log_metric("recall_macro", recall)
            mlflow.log_metric("precision_macro", precision)
            mlflow.log_param("X_train_shape", str(X_train.shape))
            mlflow.log_param("X_test_shape", str(X_test.shape))
            mlflow.log_param("y_train_shape", str(y_train.shape))
            mlflow.log_param("y_test_shape", str(y_test.shape))

            cm = confusion_matrix(y_test, preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues")
            plt.tight_layout()
            plt.savefig("confusion_matrix.png")
            plt.close()
            mlflow.log_artifact("confusion_matrix.png")

            # Como usamos el estimador sklearn de LightGBM, registramos con el flavor sklearn
            mlflow.sklearn.log_model(final_model, "model")

        LOGGER.info(
            "LightGBM entrenado. F1=%.4f, Precision=%.4f, Recall=%.4f", f1, precision, recall
        )

    meta = load_data_meta()
    train_lightgbm(meta)


dag = lightgbm_direct_dag()
