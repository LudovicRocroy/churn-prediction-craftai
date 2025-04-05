"""
Churn Model Training Script for Craft.AI Platform
--------------------------------------------------

Objective:
This script automates the training and evaluation of churn prediction models using historical customer data.
It is designed to run periodically within the Craft.AI MLOps platform to continuously evaluate new data and
produce an updated predictive model.

How it works:
1. Downloads the latest `churn.csv` file from the Craft.AI datastore.
2. Preprocesses the data by encoding categorical variables and splitting into train/test sets.
3. Trains and evaluates three machine learning models using randomized hyperparameter search:
   - XGBoostClassifier
   - RandomForestClassifier
   - CatBoostClassifier
4. Selects the best model based on accuracy, and records additional performance metrics (precision, recall, F1-score).
5. Uploads the trained model to the datastore (`CHURN_EVALUATION/MODEL`) as `best_model.joblib`.
6. Returns and logs key outputs and metrics for monitoring within Craft.AI.

Note:
- The output parameters must always be returned, even if they are `None`, to match the pipeline declaration.
- Only one model is saved (the best one) for later use in prediction pipelines.
"""

import os
import pandas as pd
import joblib
from datetime import datetime
from craft_ai_sdk import CraftAiSdk
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

def train_churn_model():
    sdk = CraftAiSdk()

    print("Téléchargement du fichier churn.csv depuis le data store.")
    file_path = "churn.csv"
    sdk.download_data_store_object("CHURN_EVALUATION/DATA/churn.csv", file_path)

    file_info = sdk.get_data_store_object_information("CHURN_EVALUATION/DATA/churn.csv")
    date_modified = file_info["last_modified"]
    print("Date fichier churn.csv :", date_modified)

    df = pd.read_csv(file_path)
    print("Préparation des données pour l'entrainement")
    df = df.dropna()
    y = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)
    X = df.drop(["customerID", "Churn"], axis=1)
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    print("Train Test Split =0.2")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {X_train.shape[0]} rows, Test size: {X_test.shape[0]} rows, Total columns: {X_train.shape[1]}")

    models = {
        "xgboost": XGBClassifier(n_jobs=-1, eval_metric="logloss"),
        "random-forest": RandomForestClassifier(n_jobs=-1),
        "cat-boost": CatBoostClassifier(thread_count=-1, silent=True)
    }

    param_grids = {
        "xgboost": {
            "max_depth": [3, 5, 7, 10],
            "n_estimators": [50, 100, 150, 200],
            "learning_rate": [0.1, 0.05, 0.01, 0.005],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "gamma": [0, 0.1, 0.5],
            "scale_pos_weight": [1, 2, 3]
        },
        "random-forest": {
            "n_estimators": [50, 100, 150],
            "max_depth": [3, 5, 7, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2, 4]
        },
        "cat-boost": {
            "iterations": [50, 100, 200],
            "depth": [3, 5, 7],
            "learning_rate": [0.1, 0.05],
            "l2_leaf_reg": [1, 3, 5]
        }
    }

    best_model = None
    best_score = 0
    best_params = {}
    best_estimator = None
    best_precision = best_recall = best_f1 = 0

    for model_name, model in models.items():
        print(f"Début de l'entrainement pour {model_name}")
        grid = RandomizedSearchCV(model, param_distributions=param_grids[model_name], n_iter=20, scoring="accuracy", cv=3, verbose=2, random_state=42, n_jobs=-1)
        grid.fit(X_train, y_train)
        print(f"Fin de l'entrainement pour {model_name}")

        y_pred = grid.predict(X_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Model: {model_name} - Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_model = model_name
            best_params = grid.best_params_
            best_estimator = grid.best_estimator_
            best_precision = precision
            best_recall = recall
            best_f1 = f1

        sdk.record_metric_value(f"{model_name}-precision", precision)
        sdk.record_metric_value(f"{model_name}-recall", recall)
        sdk.record_metric_value(f"{model_name}-f1-score", f1)

    print(f"Meilleur modèle: {best_model} avec un score de {best_score}")
    model_path = "best_model.joblib"
    joblib.dump(best_estimator, model_path)
    sdk.upload_data_store_object(model_path, "CHURN_EVALUATION/MODEL/best_model.joblib")

    return {
        "result": "Training completed",
        "best_model": best_model,
        "best_score": best_score,
        "precision": best_precision,
        "recall": best_recall,
        "f1_score": best_f1,
        "param_max_depth": best_params.get("max_depth", None),
        "param_n_estimators": best_params.get("n_estimators", None),
        "param_learning_rate": best_params.get("learning_rate", None),
    }
