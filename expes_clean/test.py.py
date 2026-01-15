import mlflow
mlflow.set_tracking_uri("file:///tmp/mlflow_test")
with mlflow.start_run():
    mlflow.log_param("test", 1)
print("Run terminé")