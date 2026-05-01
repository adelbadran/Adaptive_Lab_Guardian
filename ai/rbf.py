import numpy as np
from preprocessing import preprocess_data


class RBFNetwork:
    def __init__(self, num_centers=10, sigma=1.0):
        self.num_centers = num_centers
        self.sigma = sigma
        self.centers = None
        self.weights = None

    # ─────────────────────────────────────────────
    # Gaussian RBF Function
    # ─────────────────────────────────────────────
    def _rbf(self, x, c):
        return np.exp(-np.linalg.norm(x - c) ** 2 / (2 * self.sigma ** 2))

    # ─────────────────────────────────────────────
    # Build Interpolation Matrix (G)
    # ─────────────────────────────────────────────
    def _calculate_interpolation_matrix(self, X):
        G = np.zeros((X.shape[0], self.num_centers))

        for i, x in enumerate(X):
            for j, c in enumerate(self.centers):
                G[i, j] = self._rbf(x, c)

        return G

    # ─────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────
    def fit(self, X, y):
        # اختيار الـ centers عشوائي من البيانات
        random_idx = np.random.choice(len(X), self.num_centers, replace=False)
        self.centers = X[random_idx]

        # حساب G matrix
        G = self._calculate_interpolation_matrix(X)

        # حساب weights باستخدام pseudo-inverse
        self.weights = np.linalg.pinv(G).dot(y)

    # ─────────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────────
    def predict(self, X):
        G = self._calculate_interpolation_matrix(X)
        return G.dot(self.weights)

    # ─────────────────────────────────────────────
    # Classification Output (optional)
    # ─────────────────────────────────────────────
    def predict_classes(self, X):
        preds = self.predict(X)
        return np.round(preds).astype(int)


# =============================================================================
# TEST RUN
# =============================================================================
if __name__ == "__main__":

    print("\n Running RBF Model...\n")

    # تحميل الداتا من preprocessing
    X_train, X_test, y_train, y_test, scaler, le, feature_cols = preprocess_data(
        csv_path="data/Adaptive_Lab_Guardian.csv",
        oversample_train=False
    )

    # إنشاء الموديل
    rbf = RBFNetwork(num_centers=12, sigma=1.5)

    # تدريب
    rbf.fit(X_train, y_train)

    # توقع
    preds = rbf.predict(X_test)
    preds_classes = rbf.predict_classes(X_test)

    # تقييم
    mse = np.mean((preds - y_test) ** 2)
    accuracy = np.mean(preds_classes == y_test)

    print("MSE:", mse)
    print("Accuracy:", accuracy)

    # عرض شوية نتائج
    print("\nSample Predictions:")
    for i in range(5):
        print(f"Pred: {preds_classes[i]} | True: {y_test[i]}")