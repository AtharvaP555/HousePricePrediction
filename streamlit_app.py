import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import seaborn as sns
import os
from src.data_prep import load_and_prepare, FEATURES
from src.train import train_models

# Paths
DATA_PATH = "./data/AmesHousing.csv"
MODEL_PATH = "./models/xgb_model.pkl"


@st.cache_resource
def load_model():
    """Load trained XGBoost model, or train if not available."""
    os.makedirs("models", exist_ok=True)

    if os.path.exists(MODEL_PATH):
        # Load existing model
        model = joblib.load(MODEL_PATH)
    else:
        # Train and save XGBoost if not found
        X_train, X_test, y_train, y_test = load_and_prepare(DATA_PATH)
        models = train_models(X_train, y_train)
        model = models["XGBoost"]
        joblib.dump(model, MODEL_PATH)

    return model


def plot_feature_importance(model, feature_names, top_n=10):
    """Return a matplotlib figure showing feature importance."""
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=importance_df,
        palette="viridis",
        ax=ax
    )
    ax.set_title(f"Top {top_n} Important Features (XGBoost)")
    return fig


# User-friendly labels + tooltips
FEATURE_LABELS = {
    "Overall Qual": ("Overall Quality", "Rate overall material and finish (1 = Poor, 10 = Excellent)"),
    "Gr Liv Area": ("Living Area (sq ft)", "Above ground living area in square feet"),
    "Garage Cars": ("Garage Capacity", "Number of cars the garage can hold"),
    "Total Bsmt SF": ("Basement Area (sq ft)", "Total basement area in square feet"),
    "Year Built": ("Year Built", "The year the house was originally constructed"),
    "Full Bath": ("Full Bathrooms", "Number of full bathrooms"),
    "1st Flr SF": ("First Floor Area (sq ft)", "Size of the first floor in square feet"),
}


def main():
    st.title("🏠 House Price Prediction App")
    st.write("Enter house details to predict its price (using Ames Housing dataset).")

    # Sidebar inputs
    st.sidebar.header("House Features")
    inputs = {}
    for feature in FEATURES:
        label, help_text = FEATURE_LABELS.get(feature, (feature, ""))
        if feature in ["Overall Qual", "Full Bath", "Garage Cars"]:
            inputs[feature] = st.sidebar.slider(label, 1, 10, 5, help=help_text)
        elif feature in ["Gr Liv Area", "1st Flr SF", "Total Bsmt SF"]:
            inputs[feature] = st.sidebar.number_input(
                label, min_value=200, max_value=5000, value=1500, help=help_text
            )
        elif feature == "Year Built":
            inputs[feature] = st.sidebar.number_input(
                label, min_value=1870, max_value=2010, value=1990, help=help_text
            )

    # Load model
    model = load_model()

    # Predict
    if st.button("Predict Price"):
        X_input = pd.DataFrame([inputs])
        log_price = model.predict(X_input)[0]
        price = np.exp(log_price)

        st.success(f"Estimated House Price: **${price:,.0f}**")

        # Feature importance plot
        st.subheader("Model Insights")
        fig = plot_feature_importance(model, FEATURES, top_n=7)
        st.pyplot(fig)


if __name__ == "__main__":
    main()