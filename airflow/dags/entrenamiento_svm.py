"""
DAG: train_svm
--------------

Este DAG entrena un modelo **Support Vector Machine (SVM)** para clasificación multiclase.  
Se utiliza Optuna para búsqueda de hiperparámetros y MLflow para tracking de métricas y artefactos.  

Flujo principal:
1. Lectura de datasets desde MinIO.
2. Búsqueda de parámetros óptimos para kernel, C y gamma.
3. Entrenamiento del modelo final con los mejores hiperparámetros.
4. Log de métricas y matriz de confusión en MLflow.
5. Registro del modelo final como artefacto.

Tags: ml, optuna, minio, multiclase
"""
from airflow.decorators import dag, task
from datetime import datetime
from minio import Minio
from io import BytesIO
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.svm import SVC
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
    dag_id="train_svm",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    description="Entrena modelo SVM con Optuna y registra el mejor en MLflow",
    tags=["ml", "optuna", "minio", "multiclase"],
)
def svm_direct_dag():
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

    @task(task_id="train_svm")
    def train_svm(meta: dict) -> None:
        """Lee datos, corre Optuna (10 trials) y registra el mejor SVM en 'modelos_optimizados'."""
        bucket, keys = meta["bucket"], meta["keys"]
        X_train = _read_csv_from_minio(bucket, keys["X_train"])
        y_train = _read_csv_from_minio(bucket, keys["y_train"]).squeeze()
        X_test = _read_csv_from_minio(bucket, keys["X_test"])
        y_test = _read_csv_from_minio(bucket, keys["y_test"]).squeeze()

        LOGGER.info(
            "Shapes: X_train=%s, X_test=%s, y_train=%s, y_test=%s",
            X_train.shape, X_test.shape, y_train.shape, y_test.shape
        )

        mlflow_port = os.getenv("MLFLOW_PORT", "5000")
        mlflow.set_tracking_uri(f"http://mlflow:{mlflow_port}")

        # Optuna: experimento específico
        mlflow.set_experiment("svm_optuna")
        optuna.logging.set_verbosity(optuna.logging.INFO)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
                "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid", "linear"]),
                # gamma solo aplica a kernels no lineales; para linear se ignora sin romper
                "gamma": trial.suggest_float("gamma", 1e-4, 1e0, log=True),
            }
            model = SVC(**params)
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
        final_model = SVC(**best_params)
        final_model.fit(X_train, y_train)
        preds = final_model.predict(X_test)

        f1 = f1_score(y_test, preds, average="macro")
        recall = recall_score(y_test, preds, average="macro")
        precision = precision_score(y_test, preds, average="macro")

        mlflow.set_experiment("modelos_optimizados")
        with mlflow.start_run():
            mlflow.set_tag("model_type", "SVC")
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

            mlflow.sklearn.log_model(final_model, "model")

        LOGGER.info("SVM entrenado. F1=%.4f, Precision=%.4f, Recall=%.4f", f1, precision, recall)

    meta = load_data_meta()
    train_svm(meta)


dag = svm_direct_dag()