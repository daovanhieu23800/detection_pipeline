import mlflow
# from ultralytics import YOLO

# Configure MLflow tracking server and experiment
print('asdas')
mlflow.set_tracking_uri("http://localhost:8080")

experiment_name = "YOLOv8_Experiment"

experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    # Create experiment if it doesn't exist
    experiment_id = mlflow.create_experiment(experiment_name)

mlflow.set_experiment("YOLOv8_Experiment")

with mlflow.start_run(run_name="YOLOv8_v1") as run:
    # Optionally, load the model (not required for logging the file)
    # yolo_model = YOLO("yolov8n.pt")

    # Log the entire file as an artifact under the "model_weights" folder
    mlflow.log_artifact("yolov8n.pt", artifact_path="model_weights")
    print("Logged yolov8n.pt as an artifact.")
