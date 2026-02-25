# Blog Talking Points: Wine Quality MLOps

## Problem Statement
- Predict wine quality as a regression problem using physicochemical features.
- Move beyond model training into full experiment management and deployment readiness.

## Why MLflow
- Centralized tracking of experiments, metrics, parameters, and artifacts.
- Standardized model packaging with signatures and examples.
- Built-in model registry and stage transitions for production workflows.

## Architecture
- `main.py`: orchestration, reproducibility setup, experiment config.
- `src/data_loader.py`: data ingestion + train/test split.
- `src/models.py`: model definitions.
- `src/trainer.py`: training, tuning, logging, comparison, registration, promotion.
- `predict.py` and `serve_model.py`: production model consumption.

## Experiment Tracking
- Each model run logs:
  - hyperparameters
  - train/test split ratio
  - metrics (`r2`, `rmse`, `mae`, CV stats)
  - run tags (`project`, `author`, `stage`, `model_type`, `best_model`)
- Dataset snapshot is logged as an artifact for reproducibility.

## Model Registry
- Best model is selected automatically by CV mean score.
- Registered under `WineQualityBestModel`.
- Registry description documents selection policy.

## Production Promotion
- Latest registered version is automatically transitioned to `Production`.
- Downstream inference loads from `models:/WineQualityBestModel/Production`.

## Results
- Pipeline prints a ranked model summary and best model details.
- Production-ready model URI becomes stable for inference and serving.

## Key Learnings
- Tracking + registry significantly improve reproducibility and auditability.
- Signature and input example reduce deployment mismatch risk.
- Automated stage transition streamlines mini MLOps lifecycle for demos and blogs.
