import os
import pickle
import numpy as np
from minisom import MiniSom

class AdaptiveSomClustering:
    def __init__(self, x_dim: int = 10, y_dim: int = 10, input_len: int = 3, 
                 sigma: float = 1.0, learning_rate: float = 0.5):
        """
        Self-Organizing Map (SOM) wrapper optimized for 3D PCA features.
        Implements geometric quadrant grouping to pass steady states to Fuzzy Logic.
        """
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.input_len = input_len
        self.sigma = sigma
        self.learning_rate = learning_rate
        
        # Initialize the MiniSom architecture
        self.som = MiniSom(
            x=self.x_dim, 
            y=self.y_dim, 
            input_len=self.input_len, 
            sigma=self.sigma, 
            learning_rate=self.learning_rate
        )

    def train(self, X_pca: np.ndarray, iterations: int = 1000):
        """Trains the SOM on your PCA-reduced 3D array dataset."""
        print("[INFO] Initializing SOM weights randomly...")
        self.som.random_weights_init(X_pca)
        
        print(f"[INFO] Training SOM on {len(X_pca)} samples for {iterations} iterations...")
        self.som.train_random(X_pca, iterations)
        print("[SUCCESS] SOM training cycle complete.")

    def predict_cluster(self, x: np.ndarray) -> int:
        """
        Maps the Best Matching Unit (BMU) to one of 4 stable spatial quadrants.
        Ensures continuous relationships: Output is guaranteed to be 0, 1, 2, or 3.
        """
        # Ensure input shape is flat 1D array
        x = np.asarray(x).flatten()
        
        # Get the coordinates of the winning neuron (row, col)
        w = self.som.winner(x)
        
        # Geometric splitting based on the map center point
        is_right_half = 1 if w[1] >= (self.y_dim // 2) else 0
        is_lower_half = 1 if w[0] >= (self.x_dim // 2) else 0
        
        # Binary flag encoding -> Maps cleanly into a discrete 0-3 index
        cluster_id = is_lower_half * 2 + is_right_half
        return int(cluster_id)

    def save_model(self, filepath: str):
        """Serializes and saves the complete SOM instance to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"[SUCCESS] SOM model successfully stored at: {filepath}")

    @staticmethod
    def load_model(filepath: str) -> 'AdaptiveSomClustering':
        """Static deserializer to load your trained model in the live pipeline."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"❌ No SOM model binary found at: {filepath}")
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"[SUCCESS] SOM model loaded smoothly from: {filepath}")
        return model