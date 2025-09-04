import joblib
import numpy as np
import random
import os

def save_model(model, filepath="models/model.pkl"):
    """Save a trained model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)

def load_model(filepath="models/model.pkl"):
    """Load a trained model from disk."""
    return joblib.load(filepath)

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)