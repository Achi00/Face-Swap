# AI Face Swap and Enhancement

Python-based service that provides face swapping and enhancement capabilities using state-of-the-art AI models.

## Docker Image

Check [Docker Image](https://hub.docker.com/repository/docker/achigorgadze/face-swap-api-gpu/general) of this project. docker image includes all models and neccessary packages from get go

## Features

- **Face Swapping**: Accurately swap faces between two images using InsightFace Inswapper_128 model

- **Multiple face detection**: App detects if any of image includes multiple faces and will performa face swap on each faces from 2 images which are largest

- **Face Enhancement**: Improves face quality using GFPGAN v1.4

- **REST API**: built with FastAPI

- **GPU/CPU Support**: The API supports both CPU and NVIDIA CUDA, automatically detecting an NVIDIA GPU if available and running models on CUDA cores for improved performance, if no CUDA detected API will fall back to CPU-Only.

- **Production-Ready**: The Dockerized container uses Gunicorn, which ensures efficient request handling, improved concurrency, and better performance in production by managing multiple worker processes and gracefully handling timeouts.

## Tech Stack

- Python 3.9
- FastAPI/Flask
- ONNX Runtime
- InsightFace
- GFPGAN
- OpenCV
- PyTorch
- Docker

## How to use?

- Clone repo
- Create venv environment: `python -m venv <your_environment_name>`
- Activate venv: (on Command Prompt) `<your_environment_name>\Scripts\activate`

- Instal packages: `pip install -r requirements.txt`

- Download neccessary models

- Run: `python server.py`

Server will start on `port 5000`
