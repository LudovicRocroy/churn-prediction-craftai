# Churn Prediction Pipelines with Craft.AI

This repository contains periodic training and prediction pipelines to forecast customer churn
using XGBoost, RandomForest and CatBoost models within the Craft.AI MLOps platform.

## Files

- `src/train_churn_model.py` — trains and uploads the best model
- `src/predict_churn.py` — uses the model to predict churn daily
- `deploy_train.py` — defines the training pipeline and deployment
- `deploy_predict.py` — defines the prediction pipeline and deployment
- `requirements.txt` — all Python dependencies
- `.env` — (excluded from repo) holds Craft.AI credentials

## Deployment schedule

- Training: daily at 2 AM
- Prediction: daily at 3 AM

Rem : based on https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download dataset
