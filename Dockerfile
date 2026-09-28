FROM python:3.11-slim

WORKDIR /app

# System libraries needed by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first to keep the image small
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV YOLO_CONFIG_DIR=/tmp/ultralytics
EXPOSE 5000
CMD ["python", "app.py"]
