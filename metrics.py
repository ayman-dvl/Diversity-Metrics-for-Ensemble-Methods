"""
Diversity Metrics for Ensemble Methods
This module contains functions to calculate various diversity metrics for ensemble methods.
"""

import numpy as np
import utils as ut

### Stup functions ###
# Q-statistic
def q_statistic(a, b):
    """"
    Calculate Q-statistic between two classifiers.
    a: array of binary predictions from classifier A
    b: array of binary predictions from classifier B
    Returns: Q-statistic value
    """

    N11 = np.sum((a == 1) & (b == 1))
    N00 = np.sum((a == 0) & (b == 0))
    N10 = np.sum((a == 1) & (b == 0))
    N01 = np.sum((a == 0) & (b == 1))
    return (N11 * N00 - N10 * N01) / (N11 * N00 + N10 * N01 + 1e-10)

# Correlation
def correlation(a, b):
    """
    Calculate correlation coefficient between two classifiers.
    a, b: arrays of binary predictions from two classifiers
    Returns: correlation coefficient
    """
    N11 = np.sum((a == 1) & (b == 1))
    N00 = np.sum((a == 0) & (b == 0))
    N10 = np.sum((a == 1) & (b == 0))
    N01 = np.sum((a == 0) & (b == 1))
    numerator = N11 * N00 - N01 * N10
    denominator = np.sqrt(
        (N11 + N10) * 
        (N01 + N00) * 
        (N11 + N01) * 
        (N10 + N00)
    )+ 1e-10
    
    return numerator / denominator
# Disagreement
def disagreement(a, b):
    """
    Calculate disagreement rate between two classifiers.
    a, b: arrays of binary predictions from two classifiers
    Returns: disagreement rate
    """
    N01 = np.sum((a == 0) & (b == 1))
    N10 = np.sum((a == 1) & (b == 0))
    N = len(a)
    return (N01 + N10) / N

# Double fault
def double_fault(a, b):
    """
    Calculate double fault rate between two classifiers.
    a, b: arrays of binary predictions from two classifiers
    Returns: double fault rate
    """
    N00 = np.sum((a == 0) & (b == 0))
    N = len(a)
    return N00 / N

# Pairwise metrics
def pairwise_metrics(matrix):
    """
    Calculate pairwise metrics for a matrix of classifiers.
    matrix: 2D array where each row is a binary prediction from a classifier
    Returns: dictionary of pairwise metrics
    """
    n = matrix.shape[0]
    Qs, corrs, disagreements, double_faults = [], [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            Qs.append(q_statistic(matrix[i], matrix[j]))
            corrs.append(correlation(matrix[i], matrix[j]))
            disagreements.append(disagreement(matrix[i], matrix[j]))
            double_faults.append(double_fault(matrix[i], matrix[j]))

    return {
        "Q_statistic": np.mean(Qs),
        "correlation": np.mean(corrs),
        "disagreement": np.mean(disagreements),
        "double_fault": np.mean(double_faults)
    }

# Entropy
def entropy(matrix):
    """
    Calculate entropy for a matrix of classifiers.
    matrix: 2D array where each row is a binary prediction from a classifier
    Returns: entropy value
    """
    correct = np.sum(matrix, axis=0)
    L = matrix.shape[0]
    term= min(correct, L - correct)
    term = term / (L-(int(L/2)))
    return np.mean(term)

# Kohavi-Wolpert variance
def kw_variance(matrix):
    """
    Calculate Kohavi-Wolpert variance for a matrix of classifiers.
    matrix: 2D array where each row is a binary prediction from a classifier
    Returns: Kohavi-Wolpert variance value
    """
    mean_correct = np.mean(matrix, axis=0)
    return np.mean(mean_correct * (1 - mean_correct))

# Interrater agreement (kappa)
def kappa(matrix):  
    """
    Calculate Cohen's kappa statistic for a matrix of classifiers.
    matrix: 2D array where each row is a binary prediction from a classifier
    Returns: kappa value
    """
    p_bar = np.mean(matrix)
    p_j = np.mean(matrix, axis=0)
    P_e = np.mean(p_j ** 2 + (1 - p_j) ** 2)
    return (p_bar - P_e) / (1 - P_e + 1e-10)

# Difficulty θ
def theta(matrix):
    """
    Calculate difficulty θ value for a matrix of classifiers.
    matrix: 2D array where each row is a binary prediction from a classifier
    Returns: difficulty θ value
    """
    mean_correct = np.mean(matrix, axis=0)
    return np.var(mean_correct)

# Generalized diversity
def generalized_diversity(matrix):
    """
    Calculate generalized diversity for a matrix of classifiers.
    matrix: 2D array where each row is a binary prediction from a classifier
    Returns: generalized diversity value
    """
    m, n = matrix.shape
    incorrects = (matrix == 0).astype(int)
    shared_errors = np.dot(incorrects, incorrects.T)
    np.fill_diagonal(shared_errors, 0)
    GD = np.sum(shared_errors) / (m * (m - 1))
    return GD / n

# Coincident Failure Diversity (CFD)
def cfd(matrix):
    """
    Calculate Coincident Failure Diversity (CFD) for a matrix of classifiers.
    matrix: 2D array where each row is a binary prediction from a classifier
    Returns: Coincident Failure Diversity (CFD) value
    """
    m, _ = matrix.shape
    incorrects = (matrix == 0).astype(int)
    total_failures = np.sum(incorrects, axis=0)
    max_failures = np.max(total_failures)
    return float(1 - (max_failures / m))

# Non-pairwise metrics
def non_pairwise_metrics(matrix):
    """
    Calculate non-pairwise metrics for a matrix of classifiers.
    matrix: 2D array where each row is a binary prediction from a classifier
    Returns: dictionary of non-pairwise metrics
    """
    return {
        "entropy": entropy(matrix),
        "KW_variance": kw_variance(matrix),
        "kappa": kappa(matrix),
        "theta": theta(matrix),
        "generalized_diversity": generalized_diversity(matrix),
        "CFD": cfd(matrix)
    }

def calculate_q_statistic(M1, M2, X, y):
    """ 
    Calculate the Q-statistic between two models.
    Parameters:
    - M1: First model.
    - M2: Second model.
    - X: Feature data.
    - y: Labels.
    Returns:
    - float: Q-statistic value.
    """
    # Get oracle outputs
    oracle_M1 = ut.get_oracle_output(M1, X, y)
    oracle_M2 = ut.get_oracle_output(M2, X, y)

    # Calculate Q-statistic
    Q = q_statistic(oracle_M1, oracle_M2)
    return Q

def calculate_q_statistic_for_models(models, X, y, k= -1):
    """
    Calculate Q-statistic for an array of models.
    Parameters:
    - models: List of models.
    - X: Feature data.
    - y: Labels.
    - k: Parameter for the number of best pairs to return
    Returns:
    - dict: Dictionary of k best pairs of models by the value of Q-statistics.
        The Q-statistics are sorted in ascending order.
        Most diverse pairs are at the top of the dictionary.
    """
    n = len(models)
    model_list = list(models.values())  
    model_names = list(models.keys())
    results = {}
    for i in range(n):
        for j in range(i + 1, n):
            Q_statistic = calculate_q_statistic(model_list[i], model_list[j], X, y)
            results[f"{model_names[i]} vs {model_names[j]}"] = Q_statistic
    results = dict(list(sorted(results.items(), key=lambda item: item[1]))[:k])
    return results

def calculate_pairwise_metrics(M1, M2, X, y):
    """
    Calculate pairwise metrics between two models.
    Parameters:
    - M1: First model.
    - M2: Second model.
    - X: Feature data.
    - y: Labels.
    Returns:
    - dict: Dictionary of pairwise metrics.
    """
    # Get oracle outputs
    oracle_M1 = ut.get_oracle_output(M1, X, y)
    oracle_M2 = ut.get_oracle_output(M2, X, y)

    # Calculate metrics
    metrics = {
        "Q_statistic": q_statistic(oracle_M1, oracle_M2),
        "correlation": correlation(oracle_M1, oracle_M2),
        "disagreement": disagreement(oracle_M1, oracle_M2),
        "double_fault": double_fault(oracle_M1, oracle_M2)
    }
    return metrics

def calculate_pairwise_metrics_for_models(models, X, y, k= -1):
    """
    Calculate pairwise metrics for an array of models.
    Parameters:
    - models: List of models.
    - X: Feature data.
    - y: Labels.
    - k: Parameter for the number of best pairs to return
    Returns:
    - dict: Dictionary of k best pairs of models by the value of Q-statistics.
        The Q-statistics are sorted in ascending order.
        Most diverse pairs are
          at the top of the dictionary.
    """
    n = len(models)
    model_list = list(models.values())  
    model_names = list(models.keys())
    results = {}

    for i in range(n):
        for j in range(i + 1, n):
            metrics = calculate_pairwise_metrics(model_list[i], model_list[j], X, y)
            results[f"{model_names[i]} vs {model_names[j]}"] = metrics
    sorted_results = {
        "Q_statistic": dict(list(sorted(results.items(), key=lambda item: item[1]["Q_statistic"]))[:k]),
        "correlation": dict(list(sorted(results.items(), key=lambda item: item[1]["correlation"]))[:k]),
        "disagreement": dict(list(sorted(results.items(), key=lambda item: item[1]["disagreement"], reverse = True))[:k]),
        "double_fault": dict(list(sorted(results.items(), key=lambda item: item[1]["double_fault"]))[:k])
    }

    return sorted_results