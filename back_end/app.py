from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
import torch
from ultralytics import YOLO
import numpy as np
import cv2
import base64
from io import BytesIO
from minio import Minio
import psycopg2
import mlflow
import time
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    # Allows all origins, replace with specific origins if needed
    allow_origins=["*"],
    allow_credentials=True,
    # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_methods=["*"],
    allow_headers=["*"],  # Allows all headers
)


def load_model():
    """Load YOLO model from MLflow"""
    time.sleep(5)
    mlflow.set_tracking_uri("http://mlflow:8080")
    runs = mlflow.search_runs(search_all_experiments=True)

    if len(runs) == 0:
        time.sleep(5)
        runs = mlflow.search_runs(search_all_experiments=True)

    run_id = runs['run_id'][0]
    downloaded_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="model_weights/yolov8n.pt"
    )

    detect_model = YOLO(downloaded_path)
    id2cls = detect_model.names
    cls2id = {name: id for id, name in id2cls.items()}

    return detect_model, id2cls, cls2id


def minio_connect():
    """Establish Minio connection"""
    return Minio(
        "minio:9000",
        access_key="minioaccesskey",
        secret_key="miniosecretkey",
        secure=False
    )


def postgres_connect():
    """Establish PostgreSQL connection"""
    return psycopg2.connect(
        database="postgres_database", host="postgres",
        user="postgres_user", password="postgres_password", port=5432
    )


def create_table():
    """Create table for storing detection records"""
    cursor = postgres_client.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS file_visualization (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255) NOT NULL,
            time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            person_count INTEGER NOT NULL
        )
    """)
    postgres_client.commit()
    cursor.close()


def insert_record(filename: str, person_count: int):
    """Insert a new record into the database"""
    cursor = postgres_client.cursor()
    cursor.execute("""
        INSERT INTO file_visualization (filename, person_count) VALUES (%s, %s);
    """, (filename, person_count))
    postgres_client.commit()
    cursor.close()


def save_image_2_minio(file_name: str, file_bytes: bytes):
    """Save image to Minio"""
    bucket_name = "visualized"

    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)

    minio_client.put_object(
        bucket_name, file_name, BytesIO(file_bytes),
        length=len(file_bytes), content_type="image/jpeg"
    )


# Load resources on startup
detect_model, id2cls, cls2id = load_model()
person_cls_id = cls2id['person']
minio_client = minio_connect()
postgres_client = postgres_connect()
create_table()


@app.get("/")
def landing_page():
    return {"message": "Hello, welcome to FastAPI YOLO detection"}


@app.post("/detection")
async def detect_person(image: UploadFile = File(...)):
    """Detect persons in an uploaded image"""
    image_name = image.filename
    file_bytes = await image.read()
    file_array = np.frombuffer(file_bytes, np.uint8)
    orig_image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

    result = detect_model(orig_image)
    boxes = result[0].boxes
    person_boxes = boxes.xyxy[boxes.data[:, 5] == person_cls_id]
    n_person = person_boxes.shape[0]

    for box in person_boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(orig_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(orig_image, "person", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    _, image_buffer = cv2.imencode(".jpg", orig_image)
    image_base64 = base64.b64encode(image_buffer).decode('utf-8')

    # Save to MinIO
    save_image_2_minio(image_name, image_buffer.tobytes())

    # Save record to PostgreSQL
    insert_record(image_name, n_person)

    return JSONResponse(content={
        "visualize_image": image_base64,
        "n_person": n_person
    })


@app.get("/record")
def get_records(
    id: int = Query(None, alias="id"),  # Search by ID (optional)
    # Pagination (default: page 1)
    page_number: int = Query(1, alias="page_number", gt=0)
):
    """Fetch detection records either by ID or with pagination"""
    cursor = postgres_client.cursor()

    # If ID is provided, fetch only that record
    if id is not None:
        cursor.execute("SELECT * FROM file_visualization WHERE id = %s", (id,))
        records = cursor.fetchall()

    # Otherwise, use pagination
    else:
        page_size = 10  # Number of records per page
        offset = (page_number - 1) * page_size
        cursor.execute(
            "SELECT * FROM file_visualization LIMIT %s OFFSET %s",
            (page_size, offset)
        )
        records = cursor.fetchall()

    # Get column names
    col_names = [desc[0] for desc in cursor.description]

    # Convert records into dictionaries
    result = []
    for row in records:
        row_dict = dict(zip(col_names, row))
        if isinstance(row_dict["time"], datetime):
            # Convert datetime to string
            row_dict["time"] = row_dict["time"].isoformat()
        result.append(row_dict)

    cursor.close()

    return JSONResponse(content=result)
