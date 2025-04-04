"""
Deployment Script for Churn Model Training on Craft.AI
-------------------------------------------------------

Objective:
This script automates the deployment of a training pipeline for customer churn prediction
on the Craft.AI platform. It is intended to run periodically and retrain the model using
new data uploaded to the Craft.AI datastore.

How it works:
1. Deletes any existing pipeline named 'churntraining' (safe reset).
2. Creates a new pipeline linked to the `train_churn_model.py` script in `src/`.
3. Defines pipeline outputs such as best model hyperparameters and performance metrics.
4. Executes the pipeline once as a test run.
5. Deploys the pipeline as a periodic job scheduled daily at 2:00 AM.

Note:
- The model will be stored in the Craft.AI datastore after each training cycle.
- Ensure the `.env` file contains valid Craft.AI credentials and that dependencies are listed in `requirements.txt`.
"""



import os
from dotenv import load_dotenv
from craft_ai_sdk import CraftAiSdk
from craft_ai_sdk.io import Output, OutputDestination

load_dotenv()
sdk = CraftAiSdk()

try:
    sdk.delete_pipeline("churntraining", force_deployments_deletion=True)
except Exception as e:
    print(f"Pipeline existant ignoré : {e}")

print("Creation du pipeline")

sdk.create_pipeline(
    pipeline_name="churntraining",
    function_name="train_churn_model",
    function_path="src/train_churn_model.py",
    container_config={
        "local_folder": os.getcwd(),
        "requirements_path": "requirements.txt",
        "included_folders": ["/"]
    },
    inputs=[],
    outputs=[
        Output(name="result", data_type="string", description="Résultat de l'entraînement"),
        Output(name="param_max_depth", data_type="number", description="Paramètre max_depth"),
        Output(name="param_n_estimators", data_type="number", description="Paramètre n_estimators"),
        Output(name="param_learning_rate", data_type="number", description="Paramètre learning_rate"),
        Output(name="precision", data_type="number", description="Précision"),
        Output(name="recall", data_type="number", description="Recall"),
        Output(name="f1_score", data_type="number", description="F1 Score")
    ],
)

print("Run test du pipeline")
sdk.run_pipeline("churntraining")

print("Creation du déploiement")
sdk.create_deployment(
    pipeline_name="churntraining",
    deployment_name="churndailytraining",
    execution_rule="periodic",
    mode="elastic",
    schedule="0 2 * * *",
    outputs_mapping=[
        OutputDestination(pipeline_output_name="result", is_null=True),
        OutputDestination(pipeline_output_name="param_max_depth", is_null=True),
        OutputDestination(pipeline_output_name="param_n_estimators", is_null=True),
        OutputDestination(pipeline_output_name="param_learning_rate", is_null=True),
        OutputDestination(pipeline_output_name="precision", is_null=True),
        OutputDestination(pipeline_output_name="recall", is_null=True),
        OutputDestination(pipeline_output_name="f1_score", is_null=True)
    ]
)
