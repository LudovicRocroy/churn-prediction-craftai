
""" 
Churn Prediction Script for Craft.AI Platform
---------------------------------------------

Objective:
This script is designed to be run periodically within the Craft.AI MLOps platform. Its goal is to use the
most recently trained churn prediction model to generate churn predictions on the latest customer dataset.

How it works:
1. Downloads the latest `churn.csv` file from the Craft.AI datastore.
2. Downloads the latest trained model (`best_model.joblib`) from the datastore.
3. Preprocesses the data using the same encoding strategy used during training.
4. Performs a prediction in batch mode, converting outputs to "Yes"/"No".
5. Saves the results to `churn_predict.csv` with the original features plus a new `ChurnPrediction` column.
6. Uploads the prediction file to the `CHURN_EVALUATION/PREDICTIONS` folder in the datastore.

Note:
- The model is expected to be already trained and compatible with the preprocessed feature format.
- Categorical features must be encoded consistently with training.
"""

import pandas as pd
import joblib
from craft_ai_sdk import CraftAiSdk
from sklearn.preprocessing import LabelEncoder

def predict_churn():
    sdk = CraftAiSdk()

    # Téléchargement du fichier de données
    data_path = "churn.csv"
    sdk.download_data_store_object("CHURN_EVALUATION/DATA/churn.csv", data_path)

    # Téléchargement du modèle
    model_path = "best_model.joblib"
    sdk.download_data_store_object("CHURN_EVALUATION/MODEL/best_model.joblib", model_path)

    print("Téléchargement des données et du modèle terminé.")

    print("Chargement des données et du modèle dans la mémoire.")
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)

    # Prétraitement (identique à celui de train_churn_model)
    print("Prétraitement des données")
    X = df.drop(["customerID", "Churn"], axis=1)
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    print("Calcul des prédictions (batch mode)")
    predictions = model.predict(X)
    df["ChurnPrediction"] = ["Yes" if p == 1 else "No" for p in predictions]

    output_path = "churn_predict.csv"
    df.to_csv(output_path, index=False)

    sdk.upload_data_store_object(output_path, "CHURN_EVALUATION/PREDICTIONS/churn_predict.csv")

    return {"result": "Predictions saved"}
