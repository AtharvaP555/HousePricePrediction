from pathlib import Path
import numpy as np
import pandas as pd

from src.data_prep import load_and_prepare, FEATURES

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "AmesHousing.csv"

def test_load_and_prepare_shapes_and_columns():
    X_train, X_test, y_train, y_test, feats = load_and_prepare(str(DATA_PATH))
    # Basic sanity
    assert len(X_train) > 0 and len(X_test) > 0
    assert set(feats) == set(FEATURES)
    assert list(X_train.columns) == FEATURES
    # No NaNs
    assert int(pd.isna(X_train).sum().sum()) == 0
    assert int(pd.isna(X_test).sum().sum()) == 0
    assert int(pd.isna(y_train).sum()) == 0
    assert int(pd.isna(y_test).sum()) == 0
    # Target is log-transformed (rough range ~10–14)
    assert y_train.min() > 9.0 and y_train.max() < 15.0

def test_split_reproducible_with_seed():
    X_train1, X_test1, y_train1, y_test1, _ = load_and_prepare(str(DATA_PATH), random_state=42)
    X_train2, X_test2, y_train2, y_test2, _ = load_and_prepare(str(DATA_PATH), random_state=42)

    # Same indices means deterministic split
    assert (X_train1.index == X_train2.index).all()
    assert (X_test1.index == X_test2.index).all()
    assert np.allclose(y_train1.values, y_train2.values)
    assert np.allclose(y_test1.values, y_test2.values)
