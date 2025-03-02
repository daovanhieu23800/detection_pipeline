import asyncio
import base64
import time
from io import BytesIO
from datetime import datetime

import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from minio import Minio
import psycopg2
import mlflow

# ---- FastAPI Setup ----
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load YOLO Model ----


def load_model():
    mlflow.set_tracking_uri("http://localhost:8080")
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


detect_model, id2cls, cls2id = load_model()
person_cls_id = cls2id['person']

# ---- MinIO and Postgres Setup ----


def minio_connect():
    return Minio(
        "localhost:9000",
        access_key="minioaccesskey",
        secret_key="miniosecretkey",
        secure=False
    )


def postgres_connect():
    return psycopg2.connect(
        database="postgres_database", host="localhost",
        user="postgres_user", password="postgres_password", port=5432
    )


def create_table():
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


minio_client = minio_connect()
postgres_client = postgres_connect()
create_table()

# ---- Queue for Dynamic Batching ----
BATCH_SIZE = 4  # Adjust batch size based on your needs
queue = asyncio.Queue()
batch_results = {}

# ---- Background Task: Process Requests in Batches ----


async def batch_worker():
    while True:
        batch = []
        futures = []

        while len(batch) < BATCH_SIZE and not queue.empty():
            fut, image_name, file_bytes = await queue.get()
            batch.append((image_name, file_bytes))
            futures.append(fut)

        if batch:
            image_names, file_data = zip(*batch)
            batch_results_list = process_batch(image_names, file_data)

            # Resolve futures with results
            for future, result in zip(futures, batch_results_list):
                future.set_result(result)


def process_batch(image_names, file_data):
    """Processes a batch of images"""
    orig_images = []
    for file_bytes in file_data:
        file_array = np.frombuffer(file_bytes, np.uint8)
        orig_images.append(cv2.imdecode(file_array, cv2.IMREAD_COLOR))

    results = detect_model(orig_images)  # Batch inference

    batch_results_list = []
    for image_name, orig_image, result in zip(image_names, orig_images, results):
        boxes = result.boxes.xyxy[result.data[:, 5] == person_cls_id]
        n_person = boxes.shape[0]

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(orig_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(orig_image, "person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        _, image_buffer = cv2.imencode(".jpg", orig_image)
        image_base64 = base64.b64encode(image_buffer).decode('utf-8')

        save_image_2_minio(image_name, image_buffer.tobytes())
        insert_record(image_name, n_person)

        batch_results_list.append({
            "visualize_image": image_base64,
            "n_person": n_person
        })

    return batch_results_list

# ---- Utility Functions ----


def save_image_2_minio(file_name: str, file_bytes: bytes):
    """Save image to Minio"""
    bucket_name = "visualized"

    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)

    minio_client.put_object(
        bucket_name, file_name, BytesIO(file_bytes),
        length=len(file_bytes), content_type="image/jpeg"
    )


def insert_record(filename: str, person_count: int):
    """Insert a new record into the database"""
    cursor = postgres_client.cursor()
    cursor.execute("""INSERT INTO file_visualization (filename, person_count) VALUES (%s, %s);""",
                   (filename, person_count))
    postgres_client.commit()
    cursor.close()

# ---- API Endpoints ----


@app.get("/")
def landing_page():
    return {"message": "Hello, welcome to FastAPI YOLO detection with Dynamic Batching"}


@app.post("/detection")
async def detect_person(image: UploadFile = File(...)):
    """Queue image for detection, processed in batches"""
    file_bytes = await image.read()
    future = asyncio.Future()
    await queue.put((future, image.filename, file_bytes))
    return await future


@app.get("/record")
def get_records(page_number: int = 1):
    """Fetch detection records with pagination"""
    cursor = postgres_client.cursor()
    page_size = 10
    offset = (page_number - 1) * page_size
    cursor.execute("SELECT * FROM file_visualization LIMIT %s OFFSET %s",
                   (page_size, offset))
    records = cursor.fetchall()
    col_names = [desc[0] for desc in cursor.description]
    result = [{col: value if col != "time" else value.isoformat()
               for col, value in zip(col_names, row)} for row in records]
    cursor.close()
    return JSONResponse(content=result)

# ---- Start Background Worker ----


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(batch_worker())
