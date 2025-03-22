# # Use an official Python runtime as a parent image
# FROM python:3.9

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     gcc \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender1 \
#     libgl1-mesa-glx \
#     git \
#     && rm -rf /var/lib/apt/lists/* \
#     && pip install --upgrade pip

# # Set the working directory in the container
# WORKDIR /app

# # Copy and install requirements first
# COPY requirements.txt .
# RUN pip install --no-cache-dir --default-timeout=500 --retries=5 -r requirements.txt

# # Copy the Google Cloud credentials file into the Docker image
# COPY google.json /app/google.json

# # Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
# ENV GOOGLE_APPLICATION_CREDENTIALS=/app/google.json

# # most editable file, should be below pip install
# COPY swap_faces.py .

# # Copy the current directory contents into the container at /app
# COPY . /app

# # Make port 5000 available to the world outside this container
# EXPOSE 5000

# # Run swap_faces.py when the container launches
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--threads", "4", "swap_faces:flask_app"]

FROM python:3.9

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=500 --retries=5 -r requirements.txt

# Development: mount this file as volume
COPY server.py .
COPY predict.py .

COPY . /app
# Set environment variable to use CPU only
# ENV CUDA_VISIBLE_DEVICES="-1"

EXPOSE 5000

CMD ["gunicorn", "server:app", "-w", "3", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:5000", "--timeout", "120"]
