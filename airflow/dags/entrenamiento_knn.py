"""
DAG: train_knn
--------------

Este DAG entrena un modelo **K-Nearest Neighbors (KNN)** utilizando Optuna para
optimización de hiperparámetros y MLflow para el tracking de experimentos.

Flujo principal:
1. Carga de datos procesados desde MinIO.
2. Optimización de hiperparámetros con Optuna.
3. Entrenamiento del modelo final con los mejores parámetros.
4. Registro de métricas (F1, Precision, Recall) y artefactos en MLflow.
5. Log del modelo entrenado y matriz de confusión.

Tags: ml, optuna, minio, multiclase
"""

from airflow.decorators import dag, task
from datetime import datetime
from minio import Minio
from io import BytesIO
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.neighbors import KNeighborsClassifier
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
    """Crea un cliente de MinIO según el docker-compose actual."""
    return Minio("minio:9000", access_key="minio", secret_key="minio123", secure=False)


def _read_csv_from_minio(bucket: str, key: str) -> pd.DataFrame:
    """Lee un CSV desde MinIO y devuelve un DataFrame/Series."""
    client = _minio_client()
    obj = client.get_object(bucket, key)
    return pd.read_csv(BytesIO(obj.read()))


@dag(
    dag_id="train_knn",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    description="Entrena modelo KNN con Optuna y registra el mejor en MLflow",
    tags=["ml", "optuna", "minio", "multiclase"],
)
def knn_direct_dag():
    @task(task_id="load_data_meta")
    def load_data_meta() -> dict:
        """
        Verifica que existan los datos procesados en MinIO y devuelve solo
        las REFERENCIAS (bucket y paths). Evita empujar datos grandes por XCom.
        """
        bucket = "processed"
        keys = {
            "X_train": "X_train.csv",
            "y_train": "y_train.csv",
            "X_test": "X_test.csv",
            "y_test": "y_test.csv",
        }

        client = _minio_client()
        # Verifica existencia; si falta algo, get_object lanzará error y el task fallará "ruidoso".
        for k, key in keys.items():
            client.stat_object(bucket, key)

        # Devolvemos solo paths (XCom chico, no se loguea el dataset entero)
        meta = {"bucket": bucket, "keys": keys}
        LOGGER.info("Datos disponibles en MinIO (bucket=%s): %s", bucket, keys)
        return meta

    @task(task_id="train_knn")
    def train_knn(meta: dict) -> None:
        """
        Lee datasets desde MinIO usando las referencias de 'meta',
        ejecuta Optuna (10 trials), y registra el mejor modelo en 'modelos_optimizados'.
        """
        bucket = meta["bucket"]
        keys = meta["keys"]

        # Leer datos (acá sí traemos el contenido, pero NO lo empujamos a XCom ni lo logueamos entero)
        X_train = _read_csv_from_minio(bucket, keys["X_train"])
        y_train = _read_csv_from_minio(bucket, keys["y_train"]).squeeze()
        X_test = _read_csv_from_minio(bucket, keys["X_test"])
        y_test = _read_csv_from_minio(bucket, keys["y_test"]).squeeze()

        # Logs livianos (shapes, no arrays completos)
        LOGGER.info(
            "Shapes: X_train=%s, X_test=%s, y_train=%s, y_test=%s",
            X_train.shape,
            X_test.shape,
            y_train.shape,
            y_test.shape,
        )

        # MLflow tracking
        mlflow_port = os.getenv("MLFLOW_PORT", "5000")
        mlflow.set_tracking_uri(f"http://mlflow:{mlflow_port}")

        # =========================================
        # Optuna: experimento específico del modelo
        # =========================================
        mlflow.set_experiment("knn_optuna")
        optuna.logging.set_verbosity(optuna.logging.INFO)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_neighbors": trial.suggest_int("n_neighbors", 1, 20),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "algorithm": trial.suggest_categorical(
                    "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
                ),
            }
            model = KNeighborsClassifier(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            f1 = f1_score(y_test, preds, average="macro")

            # Log de cada trial en MLflow (nested run), con nivel INFO (no ERROR)
            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("f1_macro", f1)

            LOGGER.info("Trial %s -> f1_macro=%.4f, params=%s", trial.number, f1, params)
            return f1

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)

        best_params = study.best_params
        final_model = KNeighborsClassifier(**best_params)
        final_model.fit(X_train, y_train)
        preds = final_model.predict(X_test)

        # Métricas macro
        f1 = f1_score(y_test, preds, average="macro")
        recall = recall_score(y_test, preds, average="macro")
        precision = precision_score(y_test, preds, average="macro")

        # ================================
        # Experimento común y artefactos
        # ================================
        mlflow.set_experiment("modelos_optimizados")
        with mlflow.start_run():
            mlflow.set_tag("model_type", "KNeighborsClassifier")
            mlflow.log_params(best_params)
            mlflow.log_metric("f1_macro", f1)
            mlflow.log_metric("recall_macro", recall)
            mlflow.log_metric("precision_macro", precision)

            # Guardamos shapes como params (útil para auditoría)
            mlflow.log_param("X_train_shape", str(X_train.shape))
            mlflow.log_param("X_test_shape", str(X_test.shape))
            mlflow.log_param("y_train_shape", str(y_train.shape))
            mlflow.log_param("y_test_shape", str(y_test.shape))

            # Matriz de confusión como imagen
            cm = confusion_matrix(y_test, preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues")
            plt.tight_layout()
            plt.savefig("confusion_matrix.png")
            plt.close()
            mlflow.log_artifact("confusion_matrix.png")

            # Modelo final
            mlflow.sklearn.log_model(final_model, "model")

        LOGGER.info("Modelo entrenado. F1=%.4f, Precision=%.4f, Recall=%.4f", f1, precision, recall)

    meta = load_data_meta()
    train_knn(meta)


dag = knn_direct_dag()
