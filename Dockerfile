FROM python:3.12-slim

# prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# (1) install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# (2) copy source code, model, and data
COPY src/ /app/src/
COPY model/ /app/model/

# when running locally
RUN mkdir -p /app/data
COPY data/history.csv /app/data/

# Create the state directory
RUN mkdir -p /state && chmod 777 /state

# (3) Set PYTHONPATH so imports work
ENV PYTHONPATH=/app

# (4) set environment variables
ENV MLLP_ADDRESS=localhost:8440
ENV PAGER_ADDRESS=localhost:8441
ENV PROMETHEUS_PORT=8000

ENV MODEL_PATH=/app/model/model.onnx
ENV THRESHOLD_PATH=/app/model/threshold.txt

ENV HISTORY_PATH=/app/data/history.csv
ENV PERSISTENCE_PATH=/state/patients.pkl
ENV JOURNAL_PATH=/state/journal.jsonl
ENV SAVE_INTERVAL=100

ENV LOG_LEVEL=INFO
ENV RECONNECT=true

# (5) Expose the Prometheus metrics port
EXPOSE 8000

# (6) run
CMD ["python", "-m", "src.aki.app"]