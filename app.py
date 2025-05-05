from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import numpy as np
from metrics import calculate_pairwise_metrics, calculate_pairwise_metrics_for_models
import joblib
from tensorflow.keras.models import load_model
import os

app = FastAPI()

# In-memory storage for uploaded models
uploaded_models = {}

# Define the request body schema for pairwise metrics
class PairwiseMetricsRequest(BaseModel):
    model_a: str
    model_b: str
    X: list[list[float]]
    y: list[int]

# Define the request body schema for pairwise metrics for multiple models
class PairwiseMetricsForModelsRequest(BaseModel):
    models: list[str]  # List of model names (keys in uploaded_models)
    X: list[list[float]]
    y: list[int]
    k: int = -1  # Optional parameter for top-k results

@app.post("/upload_model/")
async def upload_model(file: UploadFile = File(...)):
    """
    Upload a model file and process it.
    """
    # Save the uploaded file temporarily
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Load the model based on file extension
    model = None
    if file.filename.endswith(".pkl"):
        model = joblib.load(file_location)
    elif file.filename.endswith(".h5"):
        model = load_model(file_location)
    else:
        os.remove(file_location)
        return {"error": "Unsupported file format. Please upload a .pkl or .h5 file."}

    # Store the model in memory
    uploaded_models[file.filename] = model

    # Clean up the temporary file
    os.remove(file_location)

    return {"message": f"Model {file.filename} uploaded and loaded successfully!"}

@app.post("/pairwise_metrics/")
def calculate_pairwise_metrics_endpoint(request: PairwiseMetricsRequest):
    """
    Calculate pairwise metrics between two models.
    """
    # Retrieve models from memory
    model_a = uploaded_models.get(request.model_a)
    model_b = uploaded_models.get(request.model_b)

    if model_a is None or model_b is None:
        return {"error": "One or both models not found. Please upload the models first."}

    # Convert input data to numpy arrays
    X = np.array(request.X)
    y = np.array(request.y)

    # Calculate pairwise metrics
    metrics = calculate_pairwise_metrics(model_a, model_b, X, y)

    return {"pairwise_metrics": metrics}

@app.post("/pairwise_metrics_for_models/")
def calculate_pairwise_metrics_for_models_endpoint(request: PairwiseMetricsForModelsRequest):
    """
    Calculate pairwise metrics for multiple models.
    """
    # Retrieve models from memory
    models = {name: uploaded_models.get(name) for name in request.models}

    if None in models.values():
        return {"error": "One or more models not found. Please upload the models first."}

    # Convert input data to numpy arrays
    X = np.array(request.X)
    y = np.array(request.y)

    # Calculate pairwise metrics for models
    metrics = calculate_pairwise_metrics_for_models(models, X, y, request.k)

    return {"pairwise_metrics_for_models": metrics}