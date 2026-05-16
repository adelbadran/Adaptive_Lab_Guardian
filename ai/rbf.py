# =========================
# RBF MODULE
# =========================

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pickle
import os


# =========================
# Gaussian Radial Basis Function
# =========================
def gaussian(x, c, sigma):
    return np.exp(-np.linalg.norm(x - c) ** 2 / (2 * sigma ** 2))


# =========================
# MODEL CLASS
# =========================
class RBFModel:
    def __init__(self, k=4, sigma=1.0):
        self.k = k
        self.sigma = sigma
        self.centers = None
        self.W = None
        self.scaler = MinMaxScaler(feature_range=(-5, 5))

    def fit(self, X):
        X_train, y_train_raw = [], []

        for i in range(1, len(X)):
            prev, curr = X[i - 1], X[i]
            overall_trend = np.mean(curr - prev)
            y_train_raw.append(overall_trend)
            X_train.append(curr)

        X_train = np.array(X_train)
        y_train_raw = np.array(y_train_raw).reshape(-1, 1)

        y_train = self.scaler.fit_transform(y_train_raw)

        kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=10)
        kmeans.fit(X_train)
        self.centers = kmeans.cluster_centers_

        Phi = np.zeros((len(X_train), self.k))
        for i in range(len(X_train)):
            for j in range(self.k):
                Phi[i, j] = gaussian(X_train[i], self.centers[j], self.sigma)

        self.W = np.linalg.pinv(Phi).dot(y_train)
        print(" RBF trained successfully")

    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        Phi = np.zeros((len(X), self.k))
        for i in range(len(X)):
            for j in range(self.k):
                Phi[i, j] = gaussian(X[i], self.centers[j], self.sigma)

        return Phi.dot(self.W)


# =========================
# PIPELINE INTEGRATION FUNCTIONS
# =========================
def step_rbf(x: np.ndarray, model: RBFModel = None):
    if model is not None:
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
            
        pred = model.predict(x)
        
        return {
            "trend": float(pred[0][0])
        }

    return {
        "trend": 0.0
    }


def train_rbf(X, path="ai/models/rbf.pkl", k=4, sigma=1.0):
    model = RBFModel(k, sigma)
    model.fit(X)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(model, f)

    return model