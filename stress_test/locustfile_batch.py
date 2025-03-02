from locust import HttpUser, task, between
import os
import time


class FastAPI9900User(HttpUser):
    host = "http://localhost:9900"  # Base URL
    wait_time = between(1, 3)

    @task
    def test_detection_9900(self):
        url = "/detection"  # Detection endpoint
        image_path = "../test_image/000000000395.jpg"

        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return

        # Open and send the image
        with open(image_path, "rb") as img_file:
            files = {"image": ("image.jpg", img_file, "image/jpeg")}
            response = self.client.post(url, files=files)

            if response.status_code == 200:
                request_id = response.json().get("request_id", "unknown")
                print(f"Request submitted to {url}. ID: {request_id}")

                # Poll for results
                result_url = f"/result/{request_id}"
                while True:
                    result_response = self.client.get(result_url)

                    if result_response.status_code == 200:
                        # Only calculate and process if response is valid
                        result_data = result_response.json()
                        print(f"Result for {request_id}: {result_data}")
                        break
                    elif result_response.status_code == 404:
                        # print(f"Request ID {request_id} not found. Exiting.")
                        break
                    else:
                        print(
                            f"Waiting for {request_id}... Status: {result_response.status_code}")
                       # time.sleep(2)  # Avoid excessive polling
            else:
                print(
                    f"Error submitting request to {url}: {response.status_code}")
