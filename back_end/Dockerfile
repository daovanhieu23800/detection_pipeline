# Use official Python image as base
FROM python:3.9-slim

# Set environment variables to prevent Python from writing bytecode and buffering stdout

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libpq-dev \
 && rm -rf /var/lib/apt/lists/*
# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the application code
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Run FastAPI using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
