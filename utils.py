"""
Utility functions.
"""

import os
import joblib
from tensorflow.python.keras.models import load_model
import metrics as me
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential

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
    if isinstance(model, Sequential):
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

def calculate_diversity_metrics_correlation(models, X, y):

    oracle_outputs = {}
    for name, model in models.items():
        oracle_outputs[name] = get_oracle_output(model, X, y)

        
    diversity_results = []

    model_names = list(oracle_outputs.keys())
    for i in range(len(model_names)):
        subset = [oracle_outputs[name] for j, name in enumerate(model_names) if j != i]
        subset_matrix = np.array(subset)
        metrics = me.non_pairwise_metrics(subset_matrix)
        pairwise_metrics = me.pairwise_metrics(subset_matrix)
        metrics.update(pairwise_metrics)
        diversity_results.append(metrics)

    df_metrics = pd.DataFrame(diversity_results)

    correlation_matrix = df_metrics.corr(method='pearson')
    return correlation_matrix

def get_least_correlated_metrics(correlation_matrix, k=5):
    """
    Get the least correlated metric pairs from a correlation matrix, removing duplicate pairs.
    
    Parameters:
        correlation_matrix (pd.DataFrame): A DataFrame containing the correlation matrix.
        k (int): The number of least correlated metric pairs to return.
    
    Returns:
        pd.DataFrame: A DataFrame containing the unique least correlated metric pairs 
                     and their correlation values.
    """
    # Flatten the correlation matrix and get the absolute values
    corr_values = correlation_matrix.abs().unstack()
    
    # Remove self-correlations (diagonal elements)
    corr_values = corr_values[corr_values.index.get_level_values(0) != corr_values.index.get_level_values(1)]
    
    # Create a set to store processed pairs
    processed_pairs = set()
    unique_pairs = []
    
    # Process pairs in ascending order of correlation
    for idx, corr in corr_values.sort_values().items():
        metric1, metric2 = idx
        # Create a frozen set to handle unordered pairs
        pair = frozenset([metric1, metric2])
        
        if pair not in processed_pairs:
            processed_pairs.add(pair)
            unique_pairs.append({
                'Metric_1': metric1,
                'Metric_2': metric2,
                'Correlation': corr
            })
            
            if len(unique_pairs) == k:
                break
    
    # Convert to DataFrame
    least_correlated = pd.DataFrame(unique_pairs)
    
    return least_correlated