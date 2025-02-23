# Use an official Python runtime as a parent image
FROM python:3.12.7-slim
#FROM python:3.9-slim
WORKDIR /mlflow/
COPY requirements.txt requirements.txt
COPY entrypoint.sh entrypoint.sh

RUN pip install --upgrade pip && \
pip install -r requirements.txt
COPY . .
# Make the entrypoint script executable
RUN chmod +x entrypoint.sh

# Expose port 8080 for the MLflow server
EXPOSE 8080

# Set the entrypoint to your script
ENTRYPOINT ["./entrypoint.sh"]
