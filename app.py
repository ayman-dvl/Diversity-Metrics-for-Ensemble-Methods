from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import numpy as np
from metrics import q_statistic
import joblib
from tensorflow.keras.models import load_model
import os

app = FastAPI()

# Define the request body schema
class QStatisticRequest(BaseModel):
    predictions_a: list[int]
    predictions_b: list[int]

# Define the endpoint
@app.post("/q_statistic/")
def calculate_q_statistic(request: QStatisticRequest):
    """
    Calculate the Q-statistic between two classifiers.
    """
    # Convert input lists to numpy arrays
    a = np.array(request.predictions_a)
    b = np.array(request.predictions_b)

    # Call the q_statistic function
    result = q_statistic(a, b)

    return {"q_statistic": result}




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
        return {"error": "Unsupported file format. Please upload a .pkl or .h5 file."}

    # Clean up the temporary file
    os.remove(file_location)

    return {"message": f"Model {file.filename} uploaded and loaded successfully!"}