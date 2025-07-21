from airflow.decorators import dag, task
from datetime import datetime
from minio import Minio
import pandas as pd
from io import BytesIO

# DAG definition
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
    def extract_from_minio():
        client = Minio(
            "minio:9000",
            access_key="minio",
            secret_key="minio123",
            secure=False
        )

        obj = client.get_object("data", "enriched_employee_dataset.csv")
        data = obj.read()
        df = pd.read_csv(BytesIO(data))
        return df.to_json()  # serialize for XCom

    @task()
    def transform_data(json_df):
        df = pd.read_json(json_df)
        df["burn_rate"] = df["spend"] / df["headcount"]
        return df.to_json()

    @task()
    def load_to_minio(json_df):
        df = pd.read_json(json_df)
        output = df.to_csv(index=False).encode("utf-8")

        client = Minio(
            "minio:9000",
            access_key="minio",
            secret_key="minio123",
            secure=False
        )

        client.put_object(
            "processed",
            "burn_rate_transformed.csv",
            data=BytesIO(output),
            length=len(output),
            content_type="application/csv"
        )

    # Define DAG execution order
    raw_data = extract_from_minio()
    transformed_data = transform_data(raw_data)
    load_to_minio(transformed_data)

etl_dag = etl_pipeline_burn_rate_minio()
