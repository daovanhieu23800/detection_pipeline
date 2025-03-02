from locust import HttpUser, task, between
import base64
import os


class StressTestUser(HttpUser):
    wait_time = between(1, 3)  # Wait time between requests

    @task(3)
    def detect_person(self):
        image_path = "../test_image/000000000395.jpg"
        with open(image_path, "rb") as img_file:
            files = {"image": (os.path.basename(
                image_path), img_file, "image/jpeg")}
            response = self.client.post("/detection", files=files)
            if response.status_code == 200:
                result = response.json()
                print(f"Detected {result['n_person']} persons in image.")
