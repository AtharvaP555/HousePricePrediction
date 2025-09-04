import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_model_performance(results_df: pd.DataFrame):
    """
    Plot RMSE and R² for different models.

    Args:
        results_df (pd.DataFrame): DataFrame with columns ["Model", "RMSE", "R²"]

    Returns:
        matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.barplot(x="Model", y="RMSE", data=results_df, ax=axes[0], hue="Model", legend=False, palette="Blues")
    axes[0].set_title("RMSE Comparison")
    axes[0].set_ylabel("RMSE")

    sns.barplot(x="Model", y="R²", data=results_df, ax=axes[1], hue="Model", legend=False, palette="Greens")
    axes[1].set_title("R² Comparison")
    axes[1].set_ylabel("R²")

    fig.tight_layout()
    return fig


def plot_predictions(y_test, y_pred, model_name: str):
    """
    Scatter plot of actual vs predicted values.

    Args:
        y_test: True target values
        y_pred: Predicted values
        model_name (str): Model name

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax, alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Predicted vs Actual ({model_name})")
    return fig


def plot_residuals(y_test, y_pred, model_name: str):
    """
    Residuals plot.

    Args:
        y_test: True target values
        y_pred: Predicted values
        model_name (str): Model name

    Returns:
        matplotlib.figure.Figure
    """
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x=y_pred, y=residuals, ax=ax, alpha=0.6)
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title(f"Residuals Plot ({model_name})")
    return fig