# Production-Ready MLOps Pipeline with MLflow
### Wine Quality Prediction | Experiment Tracking -> Model Registry -> Production Deployment

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%26%20Registry-green.svg)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

---

## Project Overview

This project demonstrates a production-style MLOps workflow for Wine Quality prediction using MLflow + scikit-learn.

Unlike basic ML projects that stop at training, this pipeline implements:

- Structured experiment tracking
- Cross-validation based model comparison
- Automated best-model selection
- Model versioning via MLflow Registry
- Automatic promotion to `Production` stage
- Reproducible tracking using SQLite backend
- Batch inference script
- REST API-based inference service

This repository simulates how ML systems are managed in real-world production environments.

---

## Why This Project?

In real-world ML systems, model training is only one part of the lifecycle.
Reproducibility, version control, traceability, and controlled deployment are essential.

This project demonstrates how to:

- Track experiments systematically
- Compare multiple candidate models
- Register and version models
- Promote best-performing models to production
- Serve production models via API

It reflects core MLOps principles used in industry systems.

---

## Architecture

```text
             +----------------------------+
             |        main.py             |
             |  orchestration + setup     |
             +-------------+--------------+
                           |
           +---------------+----------------+
           |                                |
  +--------v---------+             +--------v---------+
  | Data Loading     |             | Model Catalog    |
  | Split + Prep     |             | ML Algorithms    |
  +--------+---------+             +--------+---------+
           |                                |
           +---------------+----------------+
                           |
                   +-------v--------+
                   | Trainer Module |
                   | Train + Log +  |
                   | Compare + Reg  |
                   +-------+--------+
                           |
  +------------------------+------------------------+
  |                                                 |
  +-----v----------------+            +-----------v----------+
  | MLflow Tracking      |            | MLflow Registry      |
  | (SQLite Backend)     |            | WineQualityBestModel |
  +----------------------+            +----------------------+
  |                                                 |
  +------------------------+------------------------+
                           |
                 +----------v-----------+
                 | Inference Layer      |
                 | predict.py / Flask   |
                 +----------------------+
```

---

## Features

- SQLite-based MLflow tracking backend (`mlflow.db`)
- Structured experiment metadata with tags
- Multi-model training:
  - Linear Regression
  - Decision Tree
  - Random Forest
  - ElasticNet
  - PCA + Linear Regression
- Cross-validation-driven best model selection
- Metrics logged:
  - R2
  - RMSE
  - MAE
  - CV Mean Score
  - CV Std
- Model signature + input example logging
- Automated model registration and versioning
- Automatic promotion to `Production`
- Batch inference script
- Flask REST API for serving

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train and register models

```bash
python main.py
```

This will:

- Log experiments
- Compare models
- Register the best model
- Promote it to `Production`

### 3. Launch MLflow UI

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

---

## Model Selection Logic

1. Each model is trained with cross-validation.
2. Metrics are logged per run.
3. Best model is selected by highest `cv_mean_score`.
4. Best model is registered as `WineQualityBestModel`.
5. Model is promoted to `Production` stage automatically.

---

## Inference

### Batch prediction

```bash
python predict.py
```

Loads model from:

```text
models:/WineQualityBestModel/Production
```

### REST API

```bash
python serve_model.py
```

Health check:

```text
GET http://127.0.0.1:8000/
```

Prediction:

```text
POST http://127.0.0.1:8000/predict
```

---

## Example Output

```text
Model Performance Summary (sorted by R2):

RandomForest       R2=0.5319
LinearRegression   R2=0.4031
...

Best Model Name: RandomForest
Model Registered
Model Promoted to Production
```

---

## Resume Highlights

- Implemented automated best-model selection using cross-validation
- Integrated MLflow Tracking + Model Registry (SQLite backend)
- Designed version-controlled model lifecycle with production promotion
- Built REST API for inference using Flask
- Created reproducible experiment management pipeline

---

## Future Improvements

- Docker containerization
- CI/CD pipeline integration
- Cloud deployment (AWS / Azure / GCP)
- Model monitoring and drift detection
- Rollback strategy using registry versions

---

## Technical Blog

Full deep-dive explanation:

[https://medium.com/@byrikarthik7/from-model-training-to-production-building-a-real-mlops-pipeline-with-mlflow-1ea28e9a276a](#)

---

## License

MIT License
