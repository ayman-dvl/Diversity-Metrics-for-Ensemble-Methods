"""
Diversity Metrics for Ensemble Methods
This module contains functions to calculate various diversity metrics for ensemble methods.
"""

import numpy as np

# Q-statistic
def q_statistic(a, b):
    """"
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
    a: array of binary predictions from classifier A
    b: array of binary predictions from classifier B
    Returns: correlation coefficient
    """
    return np.corrcoef(a, b)[0, 1]

# Disagreement
def disagreement(a, b):
    """
    a: array of binary predictions from classifier A
    b: array of binary predictions from classifier B
    Returns: disagreement rate
    """
    N01 = np.sum((a == 0) & (b == 1))
    N10 = np.sum((a == 1) & (b == 0))
    N = len(a)
    return (N01 + N10) / N

# Double fault
def double_fault(a, b):
    """
    a: array of binary predictions from classifier A
    b: array of binary predictions from classifier B
    Returns: double fault rate
    """
    N00 = np.sum((a == 0) & (b == 0))
    N = len(a)
    return N00 / N

# Pairwise metrics
def pairwise_metrics(matrix):
    """
    matrix: 2D numpy array of binary predictions (shape: n_classifiers x n_test_samples)
    Returns: dictionary of pairwise metrics
    """
    n = matrix.shape[0]
    Qs, corrs, disagreements, double_faults = [], [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            a = matrix[i]
            b = matrix[j]
            Qs.append(q_statistic(a, b))
            corrs.append(correlation(a, b))
            disagreements.append(disagreement(a, b))
            double_faults.append(double_fault(a, b))

    return {
        "Q_statistic": np.mean(Qs),
        "correlation": np.mean(corrs),
        "disagreement": np.mean(disagreements),
        "double_fault": np.mean(double_faults)
    }

# Entropy
def entropy(matrix):
    """
    matrix: 2D numpy array of binary predictions (shape: n_classifiers x n_test_samples)
    Returns: entropy value
    """
    mean_correct = np.mean(matrix, axis=0)
    return -np.mean(mean_correct * np.log2(mean_correct + 1e-10) + (1 - mean_correct) * np.log2(1 - mean_correct + 1e-10))

# Kohavi-Wolpert variance
def kw_variance(matrix):
    """
    matrix: 2D numpy array of binary predictions (shape: n_classifiers x n_test_samples)
    Returns: Kohavi-Wolpert variance value
    """
    mean_correct = np.mean(matrix, axis=0)
    return np.mean(mean_correct * (1 - mean_correct))

# Interrater agreement (kappa)
def kappa(matrix):
    """
    matrix: 2D numpy array of binary predictions (shape: n_classifiers x n_test_samples)
    Returns: kappa value
    """
    p_bar = np.mean(matrix)
    p_j = np.mean(matrix, axis=0)
    P_e = np.mean(p_j ** 2 + (1 - p_j) ** 2)
    return (p_bar - P_e) / (1 - P_e + 1e-10)

# Difficulty θ
def theta(matrix):
    """
    matrix: 2D numpy array of binary predictions (shape: n_classifiers x n_test_samples)
    Returns: difficulty θ value
    """
    mean_correct = np.mean(matrix, axis=0)
    return np.var(mean_correct)

# Generalized diversity
def generalized_diversity(matrix):
    """
    matrix: 2D numpy array of binary predictions (shape: n_classifiers x n_test_samples)
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
    matrix: 2D numpy array of binary predictions (shape: n_classifiers x n_test_samples)
    Returns: Coincident Failure Diversity (CFD) value
    """
    m, _ = matrix.shape
    incorrects = (matrix == 0).astype(int)
    total_failures = np.sum(incorrects, axis=0)
    max_failures = np.max(total_failures)
    return 1 - (max_failures / m)

# Non-pairwise metrics
def non_pairwise_metrics(matrix):
    """
    matrix: 2D numpy array of binary predictions (shape: n_classifiers x n_test_samples)
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
