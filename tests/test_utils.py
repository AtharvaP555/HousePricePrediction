import os
import pytest
from sklearn.linear_model import LinearRegression
import numpy as np
from src import utils

def test_save_and_load_model(tmp_path):
    # Create a simple model
    model = LinearRegression()
    model.fit(np.array([[1], [2], [3]]), np.array([1, 2, 3]))

    # Save the model
    filepath = tmp_path / "test_model.pkl"
    utils.save_model(model, filepath)

    # Load the model
    loaded_model = utils.load_model(filepath)

    # Check if the loaded model predicts correctly
    pred = loaded_model.predict([[4]])
    assert pytest.approx(pred[0], rel=1e-2) == 4

def test_set_seed_reproducibility():
    utils.set_seed(42)
    arr1 = np.random.rand(5)
    utils.set_seed(42)
    arr2 = np.random.rand(5)

    assert np.allclose(arr1, arr2)