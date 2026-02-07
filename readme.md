
# 🚀 End-to-End Fraud Detection MLOps Pipeline

A **production-style, end-to-end MLOps project** that demonstrates how modern machine learning systems are **designed, orchestrated, tracked, and deployed** using industry-grade tools.

This project mirrors **real-world ML infrastructure** and is suitable for:
- 🎓 MSc / University practicals
- 💼 MLOps & Data Science portfolios
- 🧠 Interview system-design discussions

---

## 📌 Project Objectives

- Build a **complete ML lifecycle pipeline**
- Separate infrastructure from orchestration (production mindset)
- Track experiments and models reliably
- Store ML artifacts externally (cloud-like setup)
- Run everything locally but **cloud-ready**

---

## 🧠 What This Project Demonstrates

✔ Data ingestion and preprocessing  
✔ Model training and evaluation  
✔ Experiment tracking & metrics logging  
✔ Artifact storage (models, metrics, runs)  
✔ Workflow orchestration  
✔ Containerized infrastructure  
✔ Linux-compatible DevOps practices  

---

## 🏗️ System Architecture

```
                ┌──────────────┐
                │   Airflow    │
                │ (Orchestration)
                └──────┬───────┘
                       │
        ┌──────────────▼──────────────┐
        │      ML Training Code        │
        └──────────────┬──────────────┘
                       │
          ┌────────────▼────────────┐
          │         MLflow           │
          │ (Experiments & Registry) │
          └────────────┬────────────┘
                       │
         ┌─────────────▼─────────────┐
         │           MinIO            │
         │   (S3 Artifact Storage)    │
         └───────────────────────────┘
```

Metadata → PostgreSQL  
Artifacts → MinIO (S3-compatible)

---

## 🧰 Tech Stack

| Layer | Technology |
|-----|------------|
| Language | Python 3.10+ |
| ML | XGBoost |
| Orchestration | Apache Airflow |
| Experiment Tracking | MLflow |
| Database | PostgreSQL |
| Object Storage | MinIO |
| Messaging | Apache Kafka |
| Caching | Redis |
| Containerization | Docker & Docker Compose |
| OS Compatibility | Linux / WSL (Windows) |

---

## 📁 Project Structure

```
fraud-detection-pipeline/
├── README.md
├── Makefile
├── requirements.txt
├── fraud_detection/
│   ├── infra/          # Core infrastructure (Docker Compose)
│   ├── airflow/        # Airflow services (Docker Compose)
│   ├── src/            # ML logic
│   ├── models/         # Model artifacts
│   ├── scripts/        # Shell utilities
│   └── data/           # Dataset
```

---

## 🐧 Why WSL Is Used (Important)

### ❌ Problems with Native Windows

| Issue | Windows |
|-----|--------|
| Shell scripts (`.sh`) | ❌ CRLF issues |
| Docker Linux images | ❌ Inconsistent |
| Makefile support | ❌ Not native |
| Permission handling | ❌ Limited |
| Production similarity | ❌ Low |

### ✅ Benefits of WSL (Linux on Windows)

| Feature | WSL |
|------|-----|
| Linux kernel behavior | ✅ |
| Docker compatibility | ✅ |
| Shell scripting | ✅ |
| Makefile support | ✅ |
| Cloud parity | ✅ |

👉 **Real-world ML & DevOps systems run on Linux.**
Using WSL ensures:
- Zero cross-platform bugs
- Interview-safe setup
- Production-grade environment

---

## ⚙️ Prerequisites

- Docker Desktop
- WSL (Ubuntu recommended)
- Python 3.10+
- Git

---

## 🚀 How to Run the Project

### 1️⃣ Clone Repository

```bash
git clone <your-repo-url>
cd fraud-detection-pipeline
```

---

### 2️⃣ Create Docker Network (One-Time)

```bash
docker network create infra-net
```

---

### 3️⃣ Start Infrastructure

```bash
cd fraud_detection/infra
docker compose up -d
```

Starts:
- PostgreSQL
- MLflow
- MinIO
- Kafka
- Redis

---

### 4️⃣ Create Airflow Database

```bash
docker exec -it infra-postgres-1 psql -U mlflow -d mlflowdb
```

```sql
CREATE DATABASE airflow;
CREATE USER airflow WITH PASSWORD 'airflow';
GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;
\q
```

---

### 5️⃣ Start Airflow

```bash
cd ../airflow
docker compose up -d
```

---

## 🌐 Access Services

| Service | URL |
|------|----|
| Airflow | http://localhost:8081 |
| MLflow | http://localhost:5000 |
| MinIO | http://localhost:9001 |

---

## 🔐 Default Credentials

### Airflow
```
Username: airflow
Password: airflow
```

### MinIO
```
Username: minioadmin
Password: minioadmin
```

---

## 🪣 Purpose of MinIO

MinIO acts as a **local replacement for AWS S3**.

Used for:
- Trained model storage
- MLflow artifacts
- Metrics and logs

This enables:
- Decoupled storage
- Model versioning
- Cloud-ready architecture

---

## 🧪 How to Use the Pipeline

1. Login to Airflow
2. Enable the DAG
3. Trigger the DAG
4. Monitor tasks
5. Verify runs in MLflow
6. Inspect artifacts in MinIO

---

## 🎯 Learning Outcomes

By completing this project, you understand:
- Production ML system design
- MLOps best practices
- Linux-based DevOps workflows
- Cloud-equivalent local setups
- End-to-end ML automation

---

## 🚀 Future Enhancements

- Model serving with FastAPI
- CI/CD pipeline
- Monitoring (Prometheus + Grafana)
- Secrets management
- Cloud deployment (AWS/GCP)

---

## 👤 Author

**Rajat Pathak**  
MSc Data Science  
Focused on building **production-grade ML & MLOps systems**

---

⭐ If this project helped you, consider starring the repository.
