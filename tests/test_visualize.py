import matplotlib.figure as mpl_fig
import pandas as pd
from src.visualize import plot_model_performance, plot_predictions, plot_residuals

def test_plot_model_performance():
    df = pd.DataFrame({
        "Model": ["A", "B"],
        "RMSE": [0.1, 0.2],
        "R²": [0.9, 0.85]
    })
    fig = plot_model_performance(df)
    assert isinstance(fig, mpl_fig.Figure)

def test_plot_predictions():
    import numpy as np
    y_test = np.array([1, 2, 3, 4])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    fig = plot_predictions(y_test, y_pred, "DummyModel")
    assert isinstance(fig, mpl_fig.Figure)

def test_plot_residuals():
    import numpy as np
    y_test = np.array([1, 2, 3, 4])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    fig = plot_residuals(y_test, y_pred, "DummyModel")
    assert isinstance(fig, mpl_fig.Figure)