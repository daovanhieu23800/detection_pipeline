# Detection application pipeline
# Pipeline architecture
![Project](home_test_v2.jpg)
# Stress test
With out dynamic batching
![Testing](test_single_1000.jpg)
With dynamic batching
![Testing](test_batch_1000.jpg)
# Already done
    - Minio databse for storage image instead of store locally.
    - MLflow for tracking and logging model artifact for later. when we train multiple model.
    - Flask, BE service that can detect bounding box for person and show record that saved in Postgres.
    - For FE, using Streamlit for fast demo
    - Perform stresstest with Locust to see the performance of BE service.
    - Incorperate batching instead of single predition to reduce cost and serving more people
# Future improvement:
    - Incorperating Airflow for automative training.
    - Adding prometheus database to logging and mornitoring service.
    - Employing Dockerswarm for distributed service.
    - Using docker gpu instead of cpu
# How to run this:
    - docker compose build
    - docker compose up