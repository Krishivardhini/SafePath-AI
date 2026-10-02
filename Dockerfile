# SafePath AI Mobility Companion — Production Docker Image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app:/app/safepath" \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Install essential system dependencies for OpenCV and audio
# Note: libgl1 and libglib2.0-0 are standard for Debian 12 (bookworm)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    espeak \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pre-install CPU-only PyTorch to save ~2.5GB download and prevent OOM build failure on free tiers
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire application source code
COPY . /app

# Pre-download / verify YOLOv8n weights into image layer
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Expose default port
EXPOSE 8000

# Start SafePath FastAPI server
CMD ["python", "run_server.py"]
