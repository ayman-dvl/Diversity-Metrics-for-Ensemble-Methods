from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import numpy as np
from metrics import calculate_pairwise_metrics, calculate_pairwise_metrics_for_models
import joblib
from tensorflow.keras.models import load_model
import os
from fastapi import HTTPException
import pandas as pd

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

# In-memory storage for uploaded dataset
uploaded_dataset = {"X": None, "y": None}

@app.post("/upload_dataset/")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a dataset file in Excel or CSV format and process it.
    """
    # Save the uploaded file temporarily
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Load the dataset based on file extension
    try:
        if file.filename.endswith(".csv"):
            data = pd.read_csv(file_location)
        elif file.filename.endswith(".xlsx"):
            data = pd.read_excel(file_location)
        else:
            os.remove(file_location)
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a .csv or .xlsx file.")
    finally:
        # Clean up the temporary file
        os.remove(file_location)

    # Ensure the dataset contains 'X' and 'y' columns
    if "X" not in data.columns or "y" not in data.columns:
        raise HTTPException(status_code=400, detail="Dataset must contain 'X' and 'y' columns.")

    # Store the dataset in memory
    uploaded_dataset["X"] = data["X"].tolist()
    uploaded_dataset["y"] = data["y"].tolist()

    return {"message": "Dataset uploaded and processed successfully!"}

# Update the request bodies to fetch the uploaded dataset
@app.post("/pairwise_metrics/")
def calculate_pairwise_metrics_endpoint(request: PairwiseMetricsRequest):
    """
    Calculate pairwise metrics between two models.
    """
    if uploaded_dataset["X"] is None or uploaded_dataset["y"] is None:
        return {"error": "No dataset uploaded. Please upload a dataset first."}

    # Retrieve models from memory
    model_a = uploaded_models.get(request.model_a)
    model_b = uploaded_models.get(request.model_b)

    if model_a is None or model_b is None:
        return {"error": "One or both models not found. Please upload the models first."}

    # Convert input data to numpy arrays
    X = np.array(uploaded_dataset["X"])
    y = np.array(uploaded_dataset["y"])

    # Calculate pairwise metrics
    metrics = calculate_pairwise_metrics(model_a, model_b, X, y)

    return {"pairwise_metrics": metrics}

@app.post("/pairwise_metrics_for_models/")
def calculate_pairwise_metrics_for_models_endpoint(request: PairwiseMetricsForModelsRequest):
    """
    Calculate pairwise metrics for multiple models.
    """
    if uploaded_dataset["X"] is None or uploaded_dataset["y"] is None:
        return {"error": "No dataset uploaded. Please upload a dataset first."}

    # Retrieve models from memory
    models = {name: uploaded_models.get(name) for name in request.models}

    if None in models.values():
        return {"error": "One or more models not found. Please upload the models first."}

    # Convert input data to numpy arrays
    X = np.array(uploaded_dataset["X"])
    y = np.array(uploaded_dataset["y"])

    # Calculate pairwise metrics for models
    metrics = calculate_pairwise_metrics_for_models(models, X, y, request.k)

    return {"pairwise_metrics_for_models": metrics}
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

@app.get("/models/")
def list_uploaded_models():
    """
    List all uploaded models.
    """
    if not uploaded_models:
        return {"message": "No models have been uploaded yet."}
    return {"uploaded_models": list(uploaded_models.keys())}

@app.delete("/delete_model/{model_name}")
def delete_model(model_name: str):
    """
    Delete a model from memory.
    """
    if model_name in uploaded_models:
        del uploaded_models[model_name]
        return {"message": f"Model {model_name} deleted successfully!"}
    else:
        return {"error": f"Model {model_name} not found."}

@app.get("/dataset/")
def get_uploaded_dataset():
    """
    Get the uploaded dataset.
    """
    if uploaded_dataset["X"] is None or uploaded_dataset["y"] is None:
        return {"message": "No dataset uploaded."}
    return {
        "X": uploaded_dataset["X"],
        "y": uploaded_dataset["y"]
    }

@app.delete("/delete_dataset/")
def delete_dataset():
    """
    Delete the uploaded dataset from memory.
    """
    uploaded_dataset["X"] = None
    uploaded_dataset["y"] = None
    return {"message": "Dataset deleted successfully!"}

@app.get("/healthcheck/")
def healthcheck():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)