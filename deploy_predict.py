"""
Deployment Script for Churn Prediction on Craft.AI
--------------------------------------------------

Objective:
This script creates and deploys a pipeline on the Craft.AI platform that uses a trained churn model to generate
daily predictions based on the latest customer data.

How it works:
1. Deletes any existing pipeline named 'churnpredictor' (optional, allows clean redeployments).
2. Creates a new pipeline linked to the `predict_churn.py` function located in `src/`.
3. Defines the pipeline's output (a string indicating prediction status).
4. Runs a test execution to validate the pipeline setup.
5. Deploys the pipeline as a periodic job that runs every day at 03:00 AM.
6. Outputs are configured as non-persistent (only needed for status tracking).

Note:
- This script assumes the trained model is already available in the Craft.AI datastore.
- Ensure all dependencies are defined in `requirements.txt` and that environment variables are set in `.env`.
"""



import os
from dotenv import load_dotenv
from craft_ai_sdk import CraftAiSdk
from craft_ai_sdk.io import Output, OutputDestination

load_dotenv()
sdk = CraftAiSdk()

try:
    sdk.delete_pipeline("churnpredictor", force_deployments_deletion=True)
except Exception as e:
    print(f"Pipeline existant ignoré : {e}")

print("Création du pipeline de prédiction")

sdk.create_pipeline(
    pipeline_name="churnpredictor",
    function_name="predict_churn",
    function_path="src/predict_churn.py",
    container_config={
        "local_folder": os.getcwd(),
        "requirements_path": "requirements.txt",
        "included_folders": ["/"]
    },
    outputs=[
        Output(name="result", data_type="string", description="Résultat des prédictions")
    ],
)

print("Run test du pipeline de prédiction")
sdk.run_pipeline("churnpredictor")


print("Création du déploiement périodique de prédiction")

sdk.create_deployment(
    pipeline_name="churnpredictor",
    deployment_name="churnpredictdaily",
    execution_rule="periodic",
    mode="elastic",
    schedule="0 3 * * *",
    outputs_mapping=[
        OutputDestination(pipeline_output_name="result", is_null=True)
    ]
)
