# =============================================================================
# RBF MODULE (PRO-VERSION WITH VECTORIZATION)
# =============================================================================

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pickle
import os


# =============================================================================
# Gaussian Radial Basis Function (Vectorized for high speed)
# =============================================================================
def gaussian_vectorized(X: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    """
    حساب مصفوفة التنشيط (Activation Matrix) لـ RBF بشكل مصفوفي سريع بدون Loops.
    """
    # حساب المسافات الإقليدية المربعة بين كل العينات وكل المراكز دفعة واحدة
    # X shape: [N, Features], centers shape: [K, Features]
    dists = np.sum(X_extended := X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2) ** 2
    return np.exp(-dists / (2 * sigma ** 2))


# =============================================================================
# MODEL CLASS
# =============================================================================
class RBFModel:
    def __init__(self, k: int = 4, sigma: float = 1.0):
        """
        نموذج شبكة الدالة الشعاعية الأساسية (RBF Network) لتحليل اتجاه المستشعرات.
        مُهيئة لاستقبال مخرجات الـ PCA ثلاثية الأبعاد.
        """
        self.k = k
        self.sigma = sigma
        self.centers = None
        self.W = None
        self.scaler = MinMaxScaler(feature_range=(-5, 5))

    def fit(self, X: np.ndarray):
        """تدريب الموديل وحساب الأوزان والمراكز هندسياً"""
        X_train, y_train_raw = [], []

        # حساب الفروقات الزمنية للاتجاه (Trend Tracking)
        for i in range(1, len(X)):
            prev, curr = X[i - 1], X[i]
            overall_trend = np.mean(curr - prev)
            y_train_raw.append(overall_trend)
            X_train.append(curr)

        X_train = np.array(X_train)
        y_train_raw = np.array(y_train_raw).reshape(-1, 1)

        # تقييس مخرجات الاتجاه
        y_train = self.scaler.fit_transform(y_train_raw)

        # حساب مراكز الـ RBF باستخدام الـ KMeans الكلاسيكي
        kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=10)
        kmeans.fit(X_train)
        self.centers = kmeans.cluster_centers_

        # بناء مصفوفة التصميم الفراغي (Phi) بشكل مصفوفي فوري
        Phi = gaussian_vectorized(X_train, self.centers, self.sigma)

        # حساب المصفوفة العكسية الزائفة (Moore-Penrose pseudo-inverse) لتجنب Singular Matrices
        self.W = np.linalg.pinv(Phi).dot(y_train)
        print("[SUCCESS] تم تدريب نموذج الـ RBF بنجاح وحساب المصفوفات الوزنية.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """التنبؤ بالاتجاه القادم لعينة أو عدة عينات ممررة من الـ PCA"""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # حساب التنشيط الفوري للعينة
        Phi = gaussian_vectorized(X, self.centers, self.sigma)
        return Phi.dot(self.W)


# =============================================================================
# PIPELINE INTEGRATION FUNCTIONS
# =============================================================================
def step_rbf(x: np.ndarray, model: RBFModel = None) -> dict:
    """
    معالجة وتمرير العينات الحية في الـ Pipeline للتنبؤ بالـ Trend الفعلي.
    """
    if model is not None:
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
            
        pred = model.predict(x)
        
        # تحويل المخرجات بأمان تام إلى رقم عشري نقي (Scalar float) لمنع مشاكل الأبعاد
        trend_value = float(np.squeeze(pred))
        
        return {
            "trend": trend_value
        }

    return {
        "trend": 0.0
    }


def train_rbf(X: np.ndarray, path: str = "ai/models/rbf.pkl", k: int = 4, sigma: float = 1.0) -> RBFModel:
    """تغليف عملية بناء الموديل وحفظه بأمان باستخدام Pickle"""
    model = RBFModel(k, sigma)
    model.fit(X)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
        
    print(f"[SUCCESS] تم حفظ ملف الموديل بالكامل في: {path}")
    return model


def load_rbf(path: str = "ai/models/rbf.pkl") -> RBFModel:
    """استدعاء الموديل من الذاكرة لخط الإنتاج"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ ملف الـ RBF غير موجود في: {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"[INFO] تم استدعاء نموذج RBF بنجاح.")
    return model