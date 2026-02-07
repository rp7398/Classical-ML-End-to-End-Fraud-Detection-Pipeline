FROM python:3.10-slim

# -------------------------
# System dependencies
# -------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    netcat-openbsd \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Working directory
# -------------------------
WORKDIR /app

# -------------------------
# Python dependencies
# -------------------------
# Pin MLflow to the SAME version as your local environment
RUN pip install --no-cache-dir \
    mlflow==3.7.0 \
    psycopg2-binary \
    boto3 \
    && pip cache purge

# -------------------------
# Wait-for dependency script
# -------------------------
COPY wait-for.sh /usr/local/bin/wait-for.sh
RUN chmod +x /usr/local/bin/wait-for.sh

# -------------------------
# MLflow server port
# -------------------------
EXPOSE 5000

# -------------------------
# Start MLflow server (Postgres + MinIO ready)
# -------------------------
CMD sh -c "\
    /usr/local/bin/wait-for.sh postgres:5432 60 && \
    /usr/local/bin/wait-for.sh minio:9000 60 && \
    mlflow server \
      --host 0.0.0.0 \
      --port 5000 \
      --backend-store-uri \"${BACKEND_STORE_URI}\" \
      --default-artifact-root \"${ARTIFACT_ROOT}\" \
"
