from airflow.decorators import dag, task
from datetime import datetime
from minio import Minio
from io import BytesIO
import pandas as pd
import mlflow
import mlflow.lightgbm
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import optuna
import os
import logging

@dag(
    dag_id="train_lightgbm",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    description="Entrena modelo LightGBM",
    tags=["ml", "optuna", "minio", "multiclase"]
)
def lightgbm_direct_dag():

    @task(task_id="load_data")
    def load_data():
        client = Minio("minio:9000", access_key="minio", secret_key="minio123", secure=False)

        def read_csv(bucket, path):
            obj = client.get_object(bucket, path)
            return pd.read_csv(BytesIO(obj.read()))

        X_train = read_csv("processed", "X_train.csv")
        y_train = read_csv("processed", "y_train.csv").squeeze()
        X_test = read_csv("processed", "X_test.csv")
        y_test = read_csv("processed", "y_test.csv").squeeze()

        return {
            "X_train": X_train.to_json(),
            "y_train": y_train.to_json(),
            "X_test": X_test.to_json(),
            "y_test": y_test.to_json()
        }

    @task(task_id="train_lightgbm")
    def train_lightgbm(data: dict) -> None:
        X_train = pd.read_json(data["X_train"])
        y_train = pd.read_json(data["y_train"], typ="series")
        X_test = pd.read_json(data["X_test"])
        y_test = pd.read_json(data["y_test"], typ="series")

        mlflow_port = os.getenv("MLFLOW_PORT", "5000")
        mlflow.set_tracking_uri(f"http://mlflow:{mlflow_port}")
        mlflow.set_experiment("lightgbm_optuna")

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0)
            }
            model = LGBMClassifier(**params, random_state=42, verbose=-1)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            f1 = f1_score(y_test, preds, average="macro")

            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("f1_macro", f1)

            return f1

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)

        best_params = study.best_params
        final_model = LGBMClassifier(**best_params, random_state=42, verbose=-1)
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

            cm = confusion_matrix(y_test, preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues")
            plt.savefig("confusion_matrix.png")
            mlflow.log_artifact("confusion_matrix.png")
            plt.close()

            mlflow.lightgbm.log_model(final_model, "model")

        logging.getLogger(__name__).info(
            f"Modelo LightGBM entrenado. F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
        )

    data = load_data()
    train_lightgbm(data)

dag = lightgbm_direct_dag()