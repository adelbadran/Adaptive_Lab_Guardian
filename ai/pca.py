import os
import numpy as np
import joblib

# load pca model 
model_path = os.path.join(os.path.dirname(__file__), "models", "pca.pkl")

if os.path.exists(model_path):
    pca = joblib.load(model_path)
    print("PCA model loaded") 
else:
    pca = None
    print("PCA not found")

    # Function used main.py
def transform(x: np.ndarray) -> np.ndarray: 
    """ Input: GNN output 
        Output: 3 PCA features """
    if pca is not None:
        return pca.transform(x)
    # fallback (if PCA not trained yet)
    return x[:, :3]