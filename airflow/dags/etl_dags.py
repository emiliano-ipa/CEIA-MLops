from airflow.decorators import dag, task
from datetime import datetime
from minio import Minio
import pandas as pd
from io import BytesIO
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "plugins"))

from etl import cargar_datos, eliminar_columnas, eliminar_nulos_columna, eliminar_nulos_multiples, split_dataset, imputar_variables, clasificar_burn_rate, codificar_target, codificar_categoricas

@dag(
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    default_args={"retries": 1},
    dag_id="etl_pipeline_burn_rate_minio",
    tags=["etl", "minio", "burn-rate"]
)
def etl_pipeline_burn_rate_minio():

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
        # Load dataset
        dataset = cargar_datos(os.path.dirname(local_csv_path), os.path.basename(local_csv_path))
        dataset = eliminar_columnas(dataset, ['Employee ID', 'Date of Joining', 'Years in Company'])
        dataset = eliminar_nulos_columna(dataset, ["Burn Rate"])
        dataset = eliminar_nulos_multiples(dataset)

        # Split dataset
        X_train, X_test, y_train, y_test = split_dataset(dataset, 0.2, 'Burn Rate', 42)

        # Imputation
        vars_para_imputar = [
            'Designation', 'Resource Allocation', 'Mental Fatigue Score',
            'Work Hours per Week', 'Sleep Hours', 'Work-Life Balance Score',
            'Manager Support Score', 'Deadline Pressure Score',
            'Recognition Frequency'
        ]
        _, X_train_imp, X_test_imp = imputar_variables(X_train, X_test, vars_para_imputar, 10, 42)

        # Burn rate classification
        y_train_class, y_test_class = clasificar_burn_rate(y_train, y_test)

        # Encode target
        _, y_train_encoded, y_test_encoded = codificar_target(y_train_class, y_test_class)

        # Encode categoricals
        _, X_train_final, X_test_final = codificar_categoricas(X_train_imp, X_test_imp, ["Gender", "Company Type", "WFH Setup Available"])

        # Save processed data locally
        out_paths = {
            "X_train": "/tmp/X_train.csv",
            "X_test": "/tmp/X_test.csv",
            "y_train": "/tmp/y_train.csv",
            "y_test": "/tmp/y_test.csv"
        }

        X_train_final.to_csv(out_paths["X_train"], index=False)
        X_test_final.to_csv(out_paths["X_test"], index=False)
        y_train_encoded.to_frame().to_csv(out_paths["y_train"], index=False)
        y_test_encoded.to_frame().to_csv(out_paths["y_test"], index=False)

        return out_paths

    @task()
    def load_processed_to_minio(files_dict: dict):
        client = Minio("minio:9000", access_key="minio", secret_key="minio123", secure=False)

        for name, path in files_dict.items():
            with open(path, "rb") as f:
                data = f.read()

            client.put_object(
                "processed", f"{name}.csv",
                data=BytesIO(data),
                length=len(data),
                content_type="application/csv"
            )

    # DAG execution
    raw_file_path = extract_raw_from_minio()
    processed_files = run_etl(raw_file_path)
    load_processed_to_minio(processed_files)

etl_dag = etl_pipeline_burn_rate_minio()
