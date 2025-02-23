#!/bin/bash
set -e

echo "Starting MLflow server..."
echo "test"
mlflow server --backend-store-uri sqlite:////mlflow/mlflow.db --serve-artifacts --host 0.0.0.0 --port 8080 & 
echo "test"


echo "Waiting for MLflow server to initialize..."
# sleep 10

echo "Executing registry_model.py..."
python registry_model.py

wait
