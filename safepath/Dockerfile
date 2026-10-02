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
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    espeak \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . /app/safepath

# Pre-download YOLOv8n weights into image layer
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Expose server port
EXPOSE 8000

# Health check probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start SafePath FastAPI server
CMD ["python", "-m", "safepath.server"]
