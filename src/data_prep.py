from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES: List[str] = [
    "Overall Qual",
    "Gr Liv Area",
    "Garage Cars",
    "Total Bsmt SF",
    "Year Built",
    "Full Bath",
    "1st Flr SF",
]
TARGET: str = "SalePrice"

def load_raw(csv_path: str) -> pd.DataFrame:
    """Load the raw Ames Housing CSV."""
    return pd.read_csv(csv_path)

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the selected FEATURES + TARGET, handle missing values,
    and log-transform TARGET.
    """
    dfm = df[FEATURES + [TARGET]].copy()

    # Impute a couple of rare missing values safely with median
    for col in ["Garage Cars", "Total Bsmt SF"]:
        if dfm[col].isna().any():
            dfm[col] = dfm[col].fillna(dfm[col].median())

    # Ensure numeric types
    dfm[FEATURES] = dfm[FEATURES].apply(pd.to_numeric, errors="coerce")

    # Final NA guard (should be none for these columns)
    dfm = dfm.dropna(subset=FEATURES + [TARGET])

    # Log-transform target to reduce skew
    dfm[TARGET] = np.log1p(dfm[TARGET].astype(float))
    return dfm

def split_xy(dfm: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split into train/test."""
    X = dfm[FEATURES].astype(float)
    y = dfm[TARGET].astype(float)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def load_and_prepare(
    csv_path: str, test_size: float = 0.2, random_state: int = 42
):
    """
    One-shot convenience wrapper:
    returns X_train, X_test, y_train, y_test, FEATURES.
    """
    df = load_raw(csv_path)
    dfm = prepare_dataframe(df)
    X_train, X_test, y_train, y_test = split_xy(dfm, test_size, random_state)
    return X_train, X_test, y_train, y_test
