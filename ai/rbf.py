# =========================
# RBF MODULE
# =========================

import numpy as np
from sklearn.cluster import KMeans
import pickle
import os


# =========================
# Gaussian
# =========================
def gaussian(x, c, sigma):
    return np.exp(-np.linalg.norm(x - c) ** 2 / (2 * sigma ** 2))


# =========================
# MODEL
# =========================
class RBFModel:
    def __init__(self, k=10, sigma=1.0):
        self.k = k
        self.sigma = sigma
        self.centers = None
        self.W = None

    def fit(self, X):
        X_train, y_train = [], []

        for i in range(1, len(X)):
            prev, curr = X[i - 1], X[i]

            y_train.append([
                np.clip(curr[2] - prev[2], -5, 5),  # gas trend
                np.clip(curr[0] - prev[0], -5, 5)   # temp trend
            ])

            X_train.append(curr)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=10)
        kmeans.fit(X_train)
        self.centers = kmeans.cluster_centers_

        Phi = np.zeros((len(X_train), self.k))

        for i in range(len(X_train)):
            for j in range(self.k):
                Phi[i, j] = gaussian(X_train[i], self.centers[j], self.sigma)

        self.W = np.linalg.pinv(Phi).dot(y_train)

        print(" RBF trained")

    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        Phi = np.zeros((len(X), self.k))

        for i in range(len(X)):
            for j in range(self.k):
                Phi[i, j] = gaussian(X[i], self.centers[j], self.sigma)

        return np.clip(Phi.dot(self.W), -5, 5)


def step_rbf(x: np.ndarray, model: RBFModel = None):
    

    if model is not None:
        pred = model.predict(x)
        return {
            "gas_trend": float(pred[0][0]),
            "temp_trend": float(pred[0][1])
        }


    return {
        "gas_trend": float(x[2]),
        "temp_trend": float(x[0])
    }



def train_rbf(X, path="ai/models/rbf.pkl", k=10, sigma=1.0):
    model = RBFModel(k, sigma)
    model.fit(X)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(model, f)

    return model