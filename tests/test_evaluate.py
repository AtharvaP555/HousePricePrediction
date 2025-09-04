import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from src.evaluate import evaluate_model, compare_models

def test_evaluate_model():
    # Fake dataset
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])  # perfect linear

    # Train simple model
    model = LinearRegression().fit(X, y)

    # Evaluate
    metrics = evaluate_model(model, X, y)

    assert "RMSE" in metrics
    assert "R2" in metrics
    assert metrics["RMSE"] < 1e-6  # almost perfect
    assert abs(metrics["R2"] - 1.0) < 1e-6

def test_compare_models():
    results = {
        "Linear": {"RMSE": 0.1, "R2": 0.95},
        "RF": {"RMSE": 0.2, "R2": 0.90}
    }

    df = compare_models(results)

    # Check structure
    assert isinstance(df, pd.DataFrame)
    assert "Model" in df.columns
    assert "RMSE" in df.columns
    assert "R2" in df.columns
    assert len(df) == 2