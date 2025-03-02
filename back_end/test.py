from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import asyncio
import numpy as np
import cv2
import base64
from io import BytesIO
import torch
from ultralytics import YOLO
from datetime import datetime
from collections import defaultdict
import uuid

app = FastAPI()

# Async queue for batch processing
request_queue = asyncio.Queue()
# Dictionary to store results
results = {}

# Load YOLO model
detect_model = YOLO("yolov8n.pt")
id2cls = detect_model.names
cls2id = {name: id for id, name in id2cls.items()}
person_cls_id = cls2id['person']

# Batch size
BATCH_SIZE = 64  # You can tune this based on system performance


async def batch_worker():
    """
    Background worker that processes requests in batches.
    """
    while True:
        batch = []
        batch_request_ids = []
        batch_images = []  # Store valid images only

        # Collect requests up to BATCH_SIZE
        while len(batch) < BATCH_SIZE:
            try:
                request_id, image_bytes = await asyncio.wait_for(request_queue.get(), timeout=0.5)

                # Decode image
                file_array = np.frombuffer(image_bytes, np.uint8)
                orig_image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

                if orig_image is None:
                    print(
                        f"[ERROR] Image decoding failed for request: {request_id}")
                    results[request_id] = {"error": "Invalid image format"}
                    continue  # Skip this image

                batch.append((request_id, orig_image))
                batch_request_ids.append(request_id)
                batch_images.append(orig_image)

            except asyncio.TimeoutError:
                break  # Process whatever is collected so far

        if not batch_images:
            await asyncio.sleep(0.001)
            continue

        print(f"Processing batch of size: {len(batch_images)}")

        # Run batch inference
        results_batch = detect_model(batch_images)

        # Store results
        for (request_id, orig_image), result in zip(batch, results_batch):
            boxes = result.boxes
            person_boxes = boxes.xyxy[boxes.data[:, 5] == person_cls_id]
            n_person = person_boxes.shape[0]

            # Draw bounding boxes on the image
            for box in person_boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(orig_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(orig_image, "person", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Encode image back to base64
            _, image_buffer = cv2.imencode(".jpg", orig_image)
            image_base64 = base64.b64encode(image_buffer).decode('utf-8')

            # Store result in dictionary
            results[request_id] = {
                "visualize_image": image_base64,
                "n_person": n_person
            }

        # Remove processed requests from queue
        for _ in batch_request_ids:
            request_queue.task_done()

# Start background worker


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(batch_worker())


@app.post("/detection")
async def detect_person(image: UploadFile = File(...)):
    """
    Asynchronous API that queues the request for batch processing.
    """
    request_id = str(uuid.uuid4())  # Generate unique request ID
    file_bytes = await image.read()

    # Add request to queue
    await request_queue.put((request_id, file_bytes))

    return JSONResponse(content={"request_id": request_id, "message": "Processing started. Check status using /result"})


@app.get("/result/{request_id}")
async def get_result(request_id: str):
    """
    Retrieve the detection result for a given request ID.
    """
    if request_id in results:
        # Remove entry after serving result
        return JSONResponse(content=results.pop(request_id))
    else:
        return JSONResponse(content={"status": "Processing or invalid request ID"}, status_code=202)
