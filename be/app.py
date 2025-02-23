from flask import Flask, request, jsonify, send_file
import torch
from ultralytics import YOLO
import numpy as np
import cv2
from io import BytesIO
import base64
from minio import Minio
import psycopg2
import mlflow
import time
app = Flask(__name__)


def load_model():
    # detect_model = YOLO("yolov8n.pt")
    time.sleep(5)
    mlflow.set_tracking_uri("http://mlflow:8080")
    if len(mlflow.search_runs(search_all_experiments=True)) == 0:
        time.sleep(5)
        run_id = mlflow.search_runs(search_all_experiments=True)['run_id'][0]
    else:
        run_id = mlflow.search_runs(search_all_experiments=True)['run_id'][0]

    downloaded_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="model_weights/yolov8n.pt"
    )
    detect_model = YOLO(downloaded_path)
    id2cls = detect_model.names
    cls2id = {name: id for id, name in id2cls.items()}
    return detect_model, id2cls, cls2id


def minio_connect():
    minio_client = Minio(
        "minio:9000",
        access_key="minioaccesskey",
        secret_key="miniosecretkey",
        secure=False  # Set to True if using HTTPS
    )
    return minio_client


def postgres_connect():
    postgres_client = psycopg2.connect(
        database="postgres_database", host="postgres",
        user="postgres_user", password="postgres_password", port=5432)
    return postgres_client


def create_table(postgres_client):
    cursor = postgres_client.cursor()
    create_table_query = """
        CREATE TABLE IF NOT EXISTS file_visualization (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255) NOT NULL,
            time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            person_count INTEGER NOT NULL
        )"""
    cursor.execute(create_table_query)
    postgres_client.commit()
    cursor.close()
    # postgres_client.close()
    return


def insert_record(postgres_client, filename, person_count):
    cursor = postgres_client.cursor()
    insert_query = f"""
        INSERT INTO file_visualization (filename, person_count) VALUES (%s, %s);"""
    print(filename, person_count)
    cursor.execute(insert_query, (filename, person_count))
    postgres_client.commit()
    cursor.close()
    # postgres_client.close
    return


def save_image_2_minio(minio_client, file_name, file, length=-1):
    bucket_name = "visualized"
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)

    minio_client.put_object(
        bucket_name,
        file_name,
        BytesIO(file),
        length=len(file),
        content_type="image/jpeg"
    )


detect_model, id2cls, cls2id = load_model()
person_cls_id = cls2id['person']
minio_client = minio_connect()
postgres_client = postgres_connect()
create_table(postgres_client)


@app.route("/")
def landing_page():
    return "<p>hello<p>"


@app.route("/detection", methods=["POST"])
def detect_person():
    # image_file = request.files["file"].read()
    image_file = request.files['image']
    image_name = image_file.filename
    file_bite = np.frombuffer(image_file.read(), np.uint8)
    orig_image = cv2.imdecode(file_bite, cv2.IMREAD_COLOR)
    # orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    result = detect_model(orig_image)
    boxes = result[0].boxes
    person_boxes = boxes.xyxy[boxes.data[:, 5] == 0]
    n_person = person_boxes.shape[0]

    for box in person_boxes:
        x1, y1, x2, y2 = int(box[0].item()),  int(
            box[1].item()), int(box[2].item()), int(box[3].item())
        print(x1, y1, x2, y2)
        cv2.rectangle(orig_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(orig_image, "person", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.imwrite('test.jpg', orig_image)
    _, image_buffer = cv2.imencode(".jpg", orig_image)
    image_base64 = base64.b64encode(image_buffer).decode('utf-8')

    # return send_file(io_buffer, mimetype='image/jpg'), n_person
    save_image_2_minio(minio_client, image_name,
                       image_buffer.tobytes(), length=-1)
    insert_record(postgres_client, image_name, n_person)
    return jsonify({
        "visualize_image": image_base64,
        "n_person": n_person
    }), 200


@app.route("/record", methods=["GET"])
def GET_RECORD():
    page_number = request.args.get("page_number", default=1, type=int)
    page_size = 5
    # conn = postgres_connect()
    cursor = postgres_client.cursor()

    # Fetch all records
    # Replace with your table name
    cursor.execute(
        f"SELECT * FROM file_visualization LIMIT {page_size} OFFSET ({page_number} - 1) * {page_size}")
    records = cursor.fetchall()

    # Get column names
    col_names = [desc[0] for desc in cursor.description]

    # Convert records to list of dicts
    result = [dict(zip(col_names, row)) for row in records]

    # Close connections
    cursor.close()

    return jsonify(result), 200  # Return records as JSON
