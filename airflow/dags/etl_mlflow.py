from airflow.decorators import dag, task
from datetime import datetime
from minio import Minio
import pandas as pd
from io import BytesIO
import sys
import os
import mlflow

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "plugins"))

from etl import (
    cargar_datos,
    eliminar_columnas,
    eliminar_nulos_columna,
    eliminar_nulos_multiples,
    split_dataset,
    imputar_variables,
    clasificar_burn_rate,
    codificar_target,
    codificar_categoricas,
    standard_scaler,
    min_max_scaler,
)


@dag(
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    default_args={"retries": 1},
    dag_id="etl_mlflow",
    tags=["etl", "minio", "burn-rate", "mlflow"],
)
def etl_mlflow():
    @task()
    def extract_raw_from_minio():
        client = Minio("minio:9000", access_key="minio", secret_key="minio123", secure=False)
        obj = client.get_object("data", "enriched_employee_dataset.csv")
        data = obj.read()
        df = pd.read_csv(BytesIO(data))
        local_path = "/tmp/enriched_raw.csv"
        df.to_csv(local_path, index=False)
        return local_path

    @task()
    def run_etl(local_csv_path: str):
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("burn_rate_eda")

        with mlflow.start_run(run_name="EDA_Run"):
            # Log parameters
            mlflow.log_param("split_ratio", 0.2)
            mlflow.log_param("random_seed", 42)
            mlflow.log_param(
                "dropped_columns", ["Employee ID", "Date of Joining", "Years in Company"]
            )
            mlflow.log_param(
                "categorical_columns", ["Gender", "Company Type", "WFH Setup Available"]
            )

            # Load dataset
            dataset = cargar_datos(
                os.path.dirname(local_csv_path), os.path.basename(local_csv_path)
            )
            mlflow.log_metric("input_rows", dataset.shape[0])
            mlflow.log_metric("input_columns", dataset.shape[1])
            mlflow.log_metric("input_missing_values", dataset.isnull().sum().sum())

            # Transformations
            dataset = eliminar_columnas(
                dataset, ["Employee ID", "Date of Joining", "Years in Company"]
            )
            dataset = eliminar_nulos_columna(dataset, ["Burn Rate"])
            dataset = eliminar_nulos_multiples(dataset)
            mlflow.log_metric("post_cleaning_rows", dataset.shape[0])
            mlflow.log_metric("post_cleaning_missing_values", dataset.isnull().sum().sum())

            # Split dataset
            X_train, X_test, y_train, y_test = split_dataset(dataset, 0.2, "Burn Rate", 42)
            mlflow.log_metric("train_rows", X_train.shape[0])
            mlflow.log_metric("test_rows", X_test.shape[0])

            # Imputation
            vars_para_imputar = [
                "Designation",
                "Resource Allocation",
                "Mental Fatigue Score",
                "Work Hours per Week",
                "Sleep Hours",
                "Work-Life Balance Score",
                "Manager Support Score",
                "Deadline Pressure Score",
                "Recognition Frequency",
            ]
            _, X_train_imp, X_test_imp = imputar_variables(
                X_train, X_test, vars_para_imputar, 10, 42
            )

            # Burn rate classification
            y_train_class, y_test_class = clasificar_burn_rate(y_train, y_test)

            # Encode target
            _, y_train_encoded, y_test_encoded = codificar_target(y_train_class, y_test_class)

            # Encode categoricals
            _, X_train_final, X_test_final = codificar_categoricas(
                X_train_imp, X_test_imp, ["Gender", "Company Type", "WFH Setup Available"]
            )

            # Scale features
            _, X_train_final, X_test_final = standard_scaler(X_train_final, X_test_final)
            # OR
            # _, X_train_final, X_test_final = min_max_scaler(X_train_final, X_test_final)
            mlflow.log_param("scaler", "standard")  # o "minmax"

            # Log feature statistics
            for col in X_train_final.select_dtypes(include=["float64", "int64"]).columns:
                mlflow.log_metric(f"{col}_mean", X_train_final[col].mean())
                mlflow.log_metric(f"{col}_std", X_train_final[col].std())

            # Save processed data as artifacts
            out_paths = {
                "X_train": "/tmp/X_train.csv",
                "X_test": "/tmp/X_test.csv",
                "y_train": "/tmp/y_train.csv",
                "y_test": "/tmp/y_test.csv",
            }
            X_train_final.to_csv(out_paths["X_train"], index=False)
            X_test_final.to_csv(out_paths["X_test"], index=False)
            y_train_encoded.to_frame().to_csv(out_paths["y_train"], index=False)
            y_test_encoded.to_frame().to_csv(out_paths["y_test"], index=False)

            for name, path in out_paths.items():
                mlflow.log_artifact(path, artifact_path="processed_data")

            return out_paths

    @task()
    def load_processed_to_minio(files_dict: dict):
        client = Minio("minio:9000", access_key="minio", secret_key="minio123", secure=False)
        for name, path in files_dict.items():
            with open(path, "rb") as f:
                data = f.read()
            client.put_object(
                "processed",
                f"{name}.csv",
                data=BytesIO(data),
                length=len(data),
                content_type="application/csv",
            )

    raw_file_path = extract_raw_from_minio()
    processed_files = run_etl(raw_file_path)
    load_processed_to_minio(processed_files)


etl_dag = etl_mlflow()
