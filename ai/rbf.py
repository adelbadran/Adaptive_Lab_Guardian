import numpy as np
from preprocessing import preprocess_data
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


class RBFNetwork:
    def __init__(self, num_centers=10, sigma=1.0):
        self.num_centers = num_centers
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _rbf(self, x, c):
        return np.exp(-np.linalg.norm(x - c) ** 2 / (2 * self.sigma ** 2))

    def _build_G(self, X):
        G = np.zeros((X.shape[0], self.num_centers))
        for i, x in enumerate(X):
            for j, c in enumerate(self.centers):
                G[i, j] = self._rbf(x, c)
        return G

    def fit(self, X, y):
        # ✅ استخدام KMeans بدل العشوائي
        kmeans = KMeans(n_clusters=self.num_centers, random_state=42)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

        G = self._build_G(X)
        self.weights = np.linalg.pinv(G).dot(y)

    def predict(self, X):
        G = self._build_G(X)
        return G.dot(self.weights)







# ============================================
# 🚀 MAIN
# ============================================

if __name__ == "__main__":

    print("🚀 Running RBF Model...\n")

    X_train, X_test, y_train, y_test, *_ = preprocess_data(
        csv_path="data/Adaptive_Lab_Guardian.csv"
    )

    # 🔥 sigma ثابت مؤقتًا
    model = RBFNetwork(num_centers=12, sigma=1.5)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    preds_classes = (preds > 0.5).astype(int)

    acc = np.mean(preds_classes == y_test)

    print("🎯 Accuracy:", acc)