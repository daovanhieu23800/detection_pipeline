#!/bin/bash
set -e

# Start the MLflow server in the background
echo "Starting MLflow server..."
echo "test"
mlflow server --backend-store-uri sqlite:////mlflow/mlflow.db --serve-artifacts --host 0.0.0.0 --port 8080 & 
echo "test"

# Optionally wait for the server to be ready
# (Replace with a more robust check if needed)
echo "Waiting for MLflow server to initialize..."
# sleep 10

# Run the registry_model.py script
echo "Executing registry_model.py..."
python registry_model.py

# Wait for background processes (i.e. MLflow server) to finish
wait
