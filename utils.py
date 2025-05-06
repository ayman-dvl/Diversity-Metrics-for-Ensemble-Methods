"""
Utility functions.
"""

import os
import joblib
from tensorflow.keras.models import load_model

def get_oracle_output(model, X, y):
    """
    Generate oracle output (1=correct, 0=incorrect) for a given model.

    Parameters:
    - model: The model to evaluate.
    - X: Scaled test data.
    - y: Test labels.

    Returns:
    - numpy.ndarray: Array of oracle outputs (1=correct, 0=incorrect).
    """
    if hasattr(model, 'predict_proba'):
        y_pred = (model.predict(X) > 0.5).astype(int).flatten()
    else:
        y_pred = model.predict(X)
    return (y_pred == y).astype(int)

def load_models(models_folder):
    """
    Load models from the specified folder and return a dictionary of models.

    Args:
        models_folder (str): Path to the folder containing the models.

    Returns:
        dict: A dictionary where keys are model names and values are the loaded models.
    """
    models = {}
    for file in os.listdir(models_folder):
        model_name = file.split(".")[0]
        file_extension = file.split(".")[-1]
        if file_extension == "pkl":
            models[model_name] = joblib.load(os.path.join(models_folder, file))
            print(f"Imported sklearn model: {model_name}")
        elif file_extension == "h5":
            models[model_name] = load_model(os.path.join(models_folder, file))
            print(f"Imported keras model: {model_name}")
    return models