import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model and return metrics."""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return {"RMSE": rmse, "R2": r2, "y_pred": y_pred}

def compare_models(results):
    """
    Compare multiple models and return a dataframe.
    results: dict { model_name: {RMSE, R2} }
    """
    df = pd.DataFrame(results).T.reset_index()
    df = df.rename(columns={"index": "Model"})
    return df

def plot_model_performance(df_results):
    """Bar plots of RMSE and R² for comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.barplot(data=df_results, x="Model", y="RMSE", ax=axes[0], palette="Blues_d")
    axes[0].set_title("RMSE Comparison")
    axes[0].set_ylabel("RMSE")

    sns.barplot(data=df_results, x="Model", y="R2", ax=axes[1], palette="Greens_d")
    axes[1].set_title("R² Comparison")
    axes[1].set_ylabel("R²")

    plt.tight_layout()
    return fig

def plot_predictions(y_test, y_pred, model_name="Model"):
    """Scatter plot of actual vs predicted values."""
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    sns.lineplot(x=y_test, y=y_test, color="red", linestyle="--", label="Perfect Prediction")
    plt.xlabel("Actual SalePrice (log)")
    plt.ylabel("Predicted SalePrice (log)")
    plt.title(f"Actual vs Predicted ({model_name})")
    plt.legend()
    plt.tight_layout()
    return plt.gcf()