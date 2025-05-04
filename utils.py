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