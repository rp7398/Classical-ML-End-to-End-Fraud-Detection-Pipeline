
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
import pendulum
from datetime import timedelta
import subprocess


start_date = pendulum.datetime(2024, 6, 1, tz="UTC")

MLFLOW_URI = "http://mlflow:5000"
EXPERIMENT_NAME = "fraud_detection_experiment"

DEFAULT_ARGS = {
    "owner": "fraud-ml",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

def prepare_dataset():
    from fraud_detection.src.training.prepare_datasett import main
    main()

def train_and_eval():
    from fraud_detection.src.training.train_modell import main
    main()







def get_latest_run(filter_string):
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)

    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=filter_string,
        order_by=["start_time DESC"],
        max_results=1,
    )
    return runs[0] if runs else None

def register_candidate():
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)

    client = mlflow.tracking.MlflowClient()
    candidate = get_latest_run("tags.mlflow.runName = 'xgboost_baseline'")
    if not candidate:
        raise RuntimeError("No candidate run found")

    client.set_tag(candidate.info.run_id, "stage", "staging")

def promote_if_ok():
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)

    client = mlflow.tracking.MlflowClient()
    candidate = get_latest_run('tags.stage = "staging"')
    production = get_latest_run('tags.stage = "production"')

    candidate_loss = candidate.data.metrics.get("business_loss")
    prod_loss = production.data.metrics.get("business_loss") if production else float("inf")

    if candidate_loss < prod_loss:
        if production:
            client.set_tag(production.info.run_id, "stage", "archived")
        client.set_tag(candidate.info.run_id, "stage", "production")
    else:
        client.set_tag(candidate.info.run_id, "stage", "archived")

with DAG(
    dag_id="fraud_retrain",
    default_args=DEFAULT_ARGS,
    schedule="@daily",
    start_date=start_date,
    catchup=False,
    max_active_runs=1,
) as dag:

    t1 = PythonOperator(task_id="prepare_dataset", python_callable=prepare_dataset)
    t2 = PythonOperator(task_id="train_and_eval", python_callable=train_and_eval)
    t3 = PythonOperator(task_id="register_candidate", python_callable=register_candidate)
    t4 = PythonOperator(task_id="promote_if_ok", python_callable=promote_if_ok)

    t1 >> t2 >> t3 >> t4
