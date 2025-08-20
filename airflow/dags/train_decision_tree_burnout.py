from airflow.decorators import dag, task
from datetime import datetime
from minio import Minio
import pandas as pd
from io import BytesIO
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import sys

sys.path.append("/opt/airflow/plugins/etl")
import etl


@dag(
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    default_args={"retries": 1},
    dag_id="train_decision_tree_burnout",
    tags=["ml", "minio", "burnout", "decision-tree"],
)
def train_decision_tree_burnout():
    @task()
    def extract_data():
        client = Minio("minio:9000", access_key="minio", secret_key="minio123", secure=False)
        obj = client.get_object("data", "enriched_employee_dataset.csv")
        data = obj.read()
        df = pd.read_csv(BytesIO(data))
        return df.to_json()

    @task()
    def preprocess_and_train(json_df):
        df = pd.read_json(json_df)
        # Preprocesamiento usando funciones del plugin
        dataset = etl.eliminar_columnas(df, ["Employee ID", "Date of Joining", "Years in Company"])
        dataset = etl.eliminar_nulos_columna(dataset, ["Burn Rate"])
        dataset = etl.eliminar_nulos_multiples(dataset)
        X_train, X_test, y_train, y_test = etl.split_dataset(dataset, 0.2, "Burn Rate", 42)
        variables_para_imputar = [
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
        _, X_train_imp, X_test_imp = etl.imputar_variables(
            X_train, X_test, variables_para_imputar, 10, 42
        )
        y_train_class, y_test_class = etl.clasificar_burn_rate(y_train, y_test)
        _, y_train_enc, y_test_enc = etl.codificar_target(y_train_class, y_test_class)
        _, X_train_codif, X_test_codif = etl.codificar_categoricas(
            X_train_imp, X_test_imp, ["Gender", "Company Type", "WFH Setup Available"]
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_codif)
        X_test_scaled = scaler.transform(X_test_codif)
        # Entrenamiento DecisionTree
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train_scaled, y_train_enc)
        acc = model.score(X_test_scaled, y_test_enc)
        # MLflow tracking
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("burnout_decision_tree")
        with mlflow.start_run():
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_metric("accuracy", acc)
        return f"Modelo entrenado y logueado en MLflow. Accuracy: {acc:.4f}"

    data = extract_data()
    preprocess_and_train(data)


train_tree_dag = train_decision_tree_burnout()
