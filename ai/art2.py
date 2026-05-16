import os
import pickle
import numpy as np

class SimpleART2:
    def __init__(self, rho: float = 0.8, alpha: float = 0.1):
        """
        Adaptive Resonance Theory (ART2) engine optimized for real-time anomaly detection.
        Tailored to ingest 3D feature representations downstream from PCA.

        Parameters
        ----------
        rho   : float - Vigilance parameter bound within [0, 1]. Higher values increase sensitivity.
        alpha : float - Learning rate bound within [0, 1] for prototype update weight distribution.
        """
        self.rho = rho
        self.alpha = alpha
        self.categories = []  # Array housing stabilized prototype vectors

    def compute_anomaly_score(self, x: np.ndarray) -> float:
        """
        Processes an incoming streaming vector, maps it against established prototypes,
        executes vigilance verification, updates tracking weights, and yields an anomaly value.
        
        Input:
            x: 1D array or 3D PCA slice array
        Output:
            anomaly_score: Normalized distance rating within [0.0, 1.0]
        """
        # Enforce strict 1D array dimensionality
        x = np.asarray(x).flatten()
        
        # Calculate mathematical magnitude and extract the unit vector
        norm = np.linalg.norm(x)
        x_norm = x / norm if norm > 0 else x
        
        # Handle boundary execution: Initialize first cluster if categories are blank
        if not self.categories:
            self.categories.append(x_norm)
            return 0.0  # Initial baseline reading baseline score
            
        # Execute vectorized matrix dot-product to compute spatial cosine proximity
        similarities = [np.dot(x_norm, cat) for cat in self.categories]
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        # Clip similarity value bounds to eliminate floating-point precision leakage
        best_similarity = max(-1.0, min(1.0, best_similarity))
        
        # Scale and derive anomaly index bounds (0.0 = total convergence, 1.0 = divergence)
        anomaly_score = (1.0 - best_similarity) / 2.0
        
        # Vigilance check gate
        if best_similarity >= self.rho:
            # Resonance achieved: Shift prototype vector weight closer to input
            updated_cat = x_norm * self.alpha + self.categories[best_match_idx] * (1 - self.alpha)
            
            # Recalibrate unit scale to secure spatial vector uniformity across evaluations
            updated_norm = np.linalg.norm(updated_cat)
            self.categories[best_match_idx] = updated_cat / updated_norm if updated_norm > 0 else updated_cat
        else:
            # Resonance rejected: Spawn a fresh distinct prototype cluster in tracking space
            self.categories.append(x_norm)
            
        return float(anomaly_score)

    def is_new_pattern(self, x: np.ndarray, threshold: float = 0.35) -> bool:
        """Pipeline hook returning True when the ART2 score is anomalous."""
        return self.compute_anomaly_score(x) >= threshold

    def save_model(self, filepath: str):
        """Serializes and saves the ART2 model state to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"[SUCCESS] ART2 engine safely stored at: {filepath}")

    @staticmethod
    def load_model(filepath: str) -> 'SimpleART2':
        """Loads a pre-trained serialized ART2 model binary."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"❌ No ART2 model binary found at: {filepath}")
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"[SUCCESS] ART2 configuration restored smoothly from: {filepath}")
        return model


_default_model = SimpleART2()


def compute_anomaly_score(x: np.ndarray) -> float:
    """Module-level convenience hook used by ai.main when no pickle exists."""
    return _default_model.compute_anomaly_score(x)


def is_new_pattern(x: np.ndarray, threshold: float = 0.35) -> bool:
    """Module-level convenience hook used by ai.main."""
    return _default_model.is_new_pattern(x, threshold=threshold)
