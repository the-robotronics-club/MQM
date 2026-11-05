FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 \
    YOLO_CONFIG_DIR=/tmp/yolo MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY api ./api
COPY dashboard ./dashboard
COPY inference ./inference
COPY research ./research
COPY config ./config
COPY models/seated_classifier.pt models/queue_classifier.pt ./models/
# Bake the standard detector into the image; no first-request download.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
RUN useradd --uid 10001 --create-home app && chown -R app:app /app
USER app
ENV ATTRIBUTE_CLASSIFIERS_ENABLED=true MODEL_WEIGHTS=yolov8n.pt
EXPOSE 8000 8501
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
