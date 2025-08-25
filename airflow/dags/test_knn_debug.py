"""
DAG: test_knn_debug
-------------------

Este DAG es una versión de debug del entrenamiento KNN para identificar
problemas con el logging de artefactos en MLflow.

Incluye logging extensivo para diagnosticar problemas de conectividad
y almacenamiento de modelos.

Tags: ml, optuna, minio, multiclase, debug
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
    return Minio("s3:9000", access_key="minio", secret_key="minio123", secure=False)


def _read_csv_from_minio(bucket: str, key: str) -> pd.DataFrame:
    """Lee un CSV desde MinIO y devuelve un DataFrame/Series."""
    client = _minio_client()
    obj = client.get_object(bucket, key)
    return pd.read_csv(BytesIO(obj.read()))


@dag(
    dag_id="test_knn_debug",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    description="Debug version - Entrena modelo KNN con Optuna y registra el mejor en MLflow",
    tags=["ml", "optuna", "minio", "multiclase", "debug"],
)
def knn_debug_dag():
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
            try:
                client.stat_object(bucket, key)
                LOGGER.info("Found %s in bucket %s", key, bucket)
            except Exception as e:
                LOGGER.error("Missing file %s in bucket %s: %s", key, bucket, e)
                raise

        # Devolvemos solo paths (XCom chico, no se loguea el dataset entero)
        meta = {"bucket": bucket, "keys": keys}
        LOGGER.info("Datos disponibles en MinIO (bucket=%s): %s", bucket, keys)
        return meta

    @task(task_id="test_mlflow_connection")
    def test_mlflow_connection() -> dict:
        """
        Prueba la conexión a MLflow y reporta el estado.
        """
        # Configurar MLflow
        mlflow.set_tracking_uri("http://mlflow:5000")
        
        try:
            # Test basic connection
            tracking_uri = mlflow.get_tracking_uri()
            LOGGER.info("MLflow tracking URI: %s", tracking_uri)
            
            # Test client connection
            client = mlflow.tracking.MlflowClient()
            experiments = client.search_experiments()
            LOGGER.info("Connected to MLflow. Found %d experiments.", len(experiments))
            
            # Test S3 environment variables
            s3_endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL", "NOT SET")
            aws_access_key = os.getenv("AWS_ACCESS_KEY_ID", "NOT SET")
            aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "NOT SET")[:4] + "..." if os.getenv("AWS_SECRET_ACCESS_KEY") else "NOT SET"
            
            LOGGER.info("S3 Config - Endpoint: %s, Access Key: %s, Secret: %s", 
                       s3_endpoint, aws_access_key, aws_secret_key)
            
            return {
                "status": "success",
                "tracking_uri": tracking_uri,
                "num_experiments": len(experiments),
                "s3_endpoint": s3_endpoint
            }
            
        except Exception as e:
            LOGGER.error("MLflow connection failed: %s", e)
            return {
                "status": "failed",
                "error": str(e)
            }

    @task(task_id="train_knn_debug")
    def train_knn_debug(meta: dict, mlflow_status: dict) -> dict:
        """
        Lee datasets desde MinIO, ejecuta Optuna (5 trials para debug),
        y registra el mejor modelo en 'modelos_optimizados' con logging extensivo.
        """
        if mlflow_status["status"] != "success":
            raise RuntimeError(f"MLflow connection failed: {mlflow_status.get('error', 'Unknown error')}")
        
        bucket = meta["bucket"]
        keys = meta["keys"]

        # Leer datos (acá sí traemos el contenido, pero NO lo empujamos a XCom ni lo logueamos entero)
        try:
            X_train = _read_csv_from_minio(bucket, keys["X_train"])
            y_train = _read_csv_from_minio(bucket, keys["y_train"]).squeeze()
            X_test = _read_csv_from_minio(bucket, keys["X_test"])
            y_test = _read_csv_from_minio(bucket, keys["y_test"]).squeeze()
            LOGGER.info("Successfully loaded all datasets from MinIO")
        except Exception as e:
            LOGGER.error("Failed to load data from MinIO: %s", e)
            raise

        # Logs livianos (shapes, no arrays completos)
        LOGGER.info(
            "Shapes: X_train=%s, X_test=%s, y_train=%s, y_test=%s",
            X_train.shape,
            X_test.shape,
            y_train.shape,
            y_test.shape,
        )

        # MLflow tracking
        mlflow.set_tracking_uri("http://mlflow:5000")

        # =========================================
        # Optuna: experimento específico del modelo
        # =========================================
        try:
            mlflow.set_experiment("knn_optuna_debug")
            LOGGER.info("Set MLflow experiment: knn_optuna_debug")
        except Exception as e:
            LOGGER.error("Failed to set MLflow experiment: %s", e)
            raise

        optuna.logging.set_verbosity(optuna.logging.INFO)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_neighbors": trial.suggest_int("n_neighbors", 1, 10),  # Reduced range for debug
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "algorithm": trial.suggest_categorical(
                    "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
                ),
            }
            model = KNeighborsClassifier(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            f1 = f1_score(y_test, preds, average="macro")

            # Log de cada trial en MLflow (nested run)
            try:
                with mlflow.start_run(nested=True):
                    mlflow.log_params(params)
                    mlflow.log_metric("f1_macro", f1)
                    LOGGER.info("Trial %s logged successfully -> f1_macro=%.4f", trial.number, f1)
            except Exception as e:
                LOGGER.error("Failed to log trial %s: %s", trial.number, e)

            LOGGER.info("Trial %s -> f1_macro=%.4f, params=%s", trial.number, f1, params)
            return f1

        try:
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=5)  # Reduced for debug
            LOGGER.info("Optuna optimization completed successfully")
        except Exception as e:
            LOGGER.error("Optuna optimization failed: %s", e)
            raise

        best_params = study.best_params
        LOGGER.info("Best parameters: %s", best_params)

        # Train final model
        try:
            final_model = KNeighborsClassifier(**best_params)
            final_model.fit(X_train, y_train)
            preds = final_model.predict(X_test)
            LOGGER.info("Final model trained successfully")
        except Exception as e:
            LOGGER.error("Failed to train final model: %s", e)
            raise

        # Métricas macro
        f1 = f1_score(y_test, preds, average="macro")
        recall = recall_score(y_test, preds, average="macro")
        precision = precision_score(y_test, preds, average="macro")

        LOGGER.info("Final metrics - F1: %.4f, Precision: %.4f, Recall: %.4f", f1, precision, recall)

        # ================================
        # Experimento común y artefactos
        # ================================
        try:
            mlflow.set_experiment("modelos_optimizados")
            LOGGER.info("Set final experiment: modelos_optimizados")
        except Exception as e:
            LOGGER.error("Failed to set final experiment: %s", e)
            raise

        try:
            with mlflow.start_run() as run:
                LOGGER.info("Started MLflow run: %s", run.info.run_id)
                
                # Log basic info
                mlflow.set_tag("model_type", "KNeighborsClassifier")
                mlflow.set_tag("debug_version", "true")
                LOGGER.info("Tags set successfully")
                
                # Log parameters
                mlflow.log_params(best_params)
                LOGGER.info("Parameters logged successfully")
                
                # Log metrics
                mlflow.log_metric("f1_macro", f1)
                mlflow.log_metric("recall_macro", recall)
                mlflow.log_metric("precision_macro", precision)
                LOGGER.info("Metrics logged successfully")

                # Guardamos shapes como params (útil para auditoría)
                mlflow.log_param("X_train_shape", str(X_train.shape))
                mlflow.log_param("X_test_shape", str(X_test.shape))
                mlflow.log_param("y_train_shape", str(y_train.shape))
                mlflow.log_param("y_test_shape", str(y_test.shape))
                LOGGER.info("Shape parameters logged successfully")

                # Matriz de confusión como imagen
                try:
                    cm = confusion_matrix(y_test, preds)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                    disp.plot(cmap="Blues")
                    plt.tight_layout()
                    plt.savefig("confusion_matrix.png")
                    plt.close()
                    mlflow.log_artifact("confusion_matrix.png")
                    LOGGER.info("Confusion matrix logged successfully")
                except Exception as e:
                    LOGGER.error("Failed to log confusion matrix: %s", e)
                    # Don't raise - continue with model logging

                # Modelo final - with extensive debug
                try:
                    LOGGER.info("Attempting to log model...")
                    LOGGER.info("Model type: %s", type(final_model))
                    LOGGER.info("Model parameters: %s", final_model.get_params())
                    
                    # Try to log the model
                    mlflow.sklearn.log_model(
                        sk_model=final_model,
                        artifact_path="model",
                        registered_model_name=None  # Don't register for debug
                    )
                    LOGGER.info("Model logged successfully!")
                    
                    # Verify the model was logged by checking artifacts
                    try:
                        client = mlflow.tracking.MlflowClient()
                        artifacts = client.list_artifacts(run.info.run_id)
                        artifact_names = [art.path for art in artifacts]
                        LOGGER.info("Artifacts in run: %s", artifact_names)
                        
                        if "model" in artifact_names:
                            LOGGER.info("SUCCESS: Model artifact found in run!")
                        else:
                            LOGGER.error("ERROR: Model artifact NOT found in run!")
                            
                    except Exception as e:
                        LOGGER.error("Failed to verify artifacts: %s", e)
                        
                except Exception as e:
                    LOGGER.error("Failed to log model: %s", e)
                    LOGGER.error("Exception type: %s", type(e))
                    raise

        except Exception as e:
            LOGGER.error("MLflow run failed: %s", e)
            raise

        result = {
            "status": "success",
            "run_id": run.info.run_id,
            "f1_macro": f1,
            "precision_macro": precision,
            "recall_macro": recall,
            "best_params": best_params
        }
        
        LOGGER.info("Training completed successfully: %s", result)
        return result

    # DAG flow
    meta = load_data_meta()
    mlflow_status = test_mlflow_connection()
    result = train_knn_debug(meta, mlflow_status)


dag = knn_debug_dag()