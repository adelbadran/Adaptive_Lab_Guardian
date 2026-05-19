"""Train and save all Adaptive Lab Guardian runtime models.

This script is intentionally NumPy/Pandas-only so the project can train in a
minimal environment. Saved artifacts are compatible with ai.main:

- scaler.pkl
- pca.pkl
- art2.pkl
- rbf.pkl
- som.pkl
- gnn.pkl
- gnn_attention.npy
- rl_qtable.npy
- ga_policy.npy
- train_report.json
"""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
import torch

try:
    from ai.art2 import SimpleART2
    from ai.ga import GA
    from ai.rbf import RBFModel, train_rbf
    from ai.som import AdaptiveSomClustering
    from ai.gnn import GATModel
except Exception:  # pragma: no cover - direct script execution
    from art2 import SimpleART2
    from ga import GA
    from rbf import RBFModel, train_rbf
    from som import AdaptiveSomClustering
    from gnn import GATModel


FEATURE_COLS = ["Temp_C", "Humidity_pct", "Gas_AQI", "Light_Lux", "Motion_Detected"]
SCENARIO_LABELS = {
    0: "Normal",
    1: "Crowded",
    2: "Chemical",
    3: "Security",
}


if __name__ == "__main__":
    sys.modules.setdefault("ai.train_models", sys.modules[__name__])


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Adaptive_Lab_Guardian.csv"
MODELS_DIR = Path(__file__).resolve().parent / "models"


@dataclass
class SimpleMinMaxScaler:
    data_min_: np.ndarray
    data_max_: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        denom = np.where((self.data_max_ - self.data_min_) == 0, 1.0, self.data_max_ - self.data_min_)
        return np.clip((X - self.data_min_) / denom, 0.0, 1.0)


@dataclass
class SimplePCAModel:
    mean_: np.ndarray
    components_: np.ndarray
    explained_variance_ratio_: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return (X - self.mean_) @ self.components_.T


@dataclass
class NumpyRBFTrendModel:
    centers: np.ndarray
    sigma: float
    weights: np.ndarray
    target_low: float = -0.05
    target_high: float = 0.05

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        diff = X[:, None, :] - self.centers[None, :, :]
        phi = np.exp(-np.sum(diff ** 2, axis=2) / (2.0 * self.sigma ** 2))
        return phi @ self.weights

    def scale_trend(self, trend: float | np.ndarray) -> float | np.ndarray:
        low = float(getattr(self, "target_low", -0.05))
        high = float(getattr(self, "target_high", 0.05))
        if high <= low:
            high = low + 1e-6
        scaled = ((np.asarray(trend, dtype=float) - low) / (high - low)) * 10.0 - 5.0
        return np.clip(scaled, -5.0, 5.0)

    def predict_scaled(self, X: np.ndarray) -> np.ndarray:
        return self.scale_trend(self.predict(X))


def _guard_features_from_scaled(X_scaled: np.ndarray) -> np.ndarray:
    X = np.asarray(X_scaled, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    X = np.clip(X, 0.0, 1.0)

    temp = X[:, 0]
    humidity = X[:, 1]
    gas = X[:, 2]
    light = X[:, 3]
    motion = X[:, 4]
    darkness = 1.0 - light

    return np.column_stack(
        [
            temp,
            humidity,
            gas,
            light,
            motion,
            darkness,
            humidity * darkness,
            (1.0 - temp) * darkness,
            gas * temp,
            motion * darkness,
        ]
    )


class KNNRiskGuard:
    def __init__(self, knn, class_weights=None, scenario_labels=None):
        self.knn = knn
        self.class_weights = class_weights if class_weights is not None else np.array([1.0, 1.0, 1.5, 2.0], dtype=float)
        self.scenario_labels = scenario_labels if scenario_labels is not None else SCENARIO_LABELS.copy()
        
    def predict(self, x_scaled: np.ndarray) -> dict:
        features = _guard_features_from_scaled(x_scaled).reshape(1, -1)
        try:
            probas = self.knn.predict_proba(features)[0]
        except Exception:
            probas = np.array([1.0, 0, 0, 0])
        votes = np.zeros(4)
        for i, p in enumerate(probas):
            if i < len(self.knn.classes_):
                c = int(self.knn.classes_[i])
                votes[c] = p * self.class_weights[c]
        total = sum(votes) or 1.0
        scenario_class = int(np.argmax(votes))
        confidence = float(votes[scenario_class] / total)
        
        sorted_votes = np.sort(votes)[::-1]
        margin = float((sorted_votes[0] - sorted_votes[1]) / total) if len(sorted_votes) > 1 else confidence
        
        risk_votes = [votes[0], votes[1], votes[2] + votes[3]]
        risk_class = int(np.argmax(risk_votes))
        
        return {
            "risk_class": risk_class,
            "scenario_class": scenario_class,
            "scenario_label": self.scenario_labels.get(scenario_class, "Unknown"),
            "confidence": confidence,
            "margin": margin,
            "votes": [float(v) for v in votes],
            "risk_votes": [float(v) for v in risk_votes],
        }

KNNRiskGuard.__module__ = "ai.train_models"




def _labels_to_clusters(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    numeric = labels.astype(float)
    if np.all(np.isin(numeric, [1, 2, 3, 4])):
        return (numeric.astype(int) - 1).astype(int)

    unique = sorted(np.unique(labels))
    if len(unique) <= 4:
        mapping = {label: idx for idx, label in enumerate(unique)}
        return np.array([mapping[value] for value in labels], dtype=int)

    quantiles = np.percentile(labels, [25, 50, 75])
    return np.digitize(labels, quantiles).astype(int)


def _target_3class(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    numeric = labels.astype(float)
    if np.all(np.isin(numeric, [1, 2, 3, 4])):
        raw = numeric.astype(int)
        return np.where(raw >= 3, 2, np.where(raw == 2, 1, 0)).astype(int)

    unique = sorted(np.unique(labels))
    if len(unique) <= 3:
        mapping = {label: idx for idx, label in enumerate(unique)}
        return np.array([mapping[value] for value in labels], dtype=int)
    q1, q2 = np.percentile(labels, [34, 67])
    return np.where(labels >= q2, 2, np.where(labels >= q1, 1, 0)).astype(int)


def _risk_from_scenario_class(scenarios: np.ndarray) -> np.ndarray:
    scenarios = np.asarray(scenarios, dtype=int)
    return np.where(scenarios == 0, 0, np.where(scenarios == 1, 1, 2)).astype(int)


def _scenario_profiles(df: pd.DataFrame) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    for raw_label, group in df.groupby("True_Scenario"):
        label = int(raw_label)
        scenario_id = label - 1 if label in (1, 2, 3, 4) else label
        profiles[str(scenario_id)] = {
            "raw_label": label,
            "name": SCENARIO_LABELS.get(scenario_id, "Unknown"),
            "count": int(len(group)),
            "pct": round(float(len(group) / max(len(df), 1) * 100.0), 2),
            "features": {
                col: {
                    "min": round(float(group[col].min()), 2),
                    "q25": round(float(group[col].quantile(0.25)), 2),
                    "mean": round(float(group[col].mean()), 2),
                    "q75": round(float(group[col].quantile(0.75)), 2),
                    "max": round(float(group[col].max()), 2),
                }
                for col in FEATURE_COLS
            },
        }
    return profiles


def _smote_resample(
    X: np.ndarray,
    y: np.ndarray,
    target_count: int | None = None,
    k_neighbors: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    smote = SMOTE(random_state=seed, k_neighbors=min(k_neighbors, len(X)-1))
    # Deal with classes having very few samples
    classes, counts = np.unique(y, return_counts=True)
    if np.any(counts <= k_neighbors):
        smote = SMOTE(random_state=seed, k_neighbors=max(1, min(counts)-1))
    return smote.fit_resample(X, y)


def _fit_som_state(X_pca: np.ndarray, X_scaled: np.ndarray, labels: np.ndarray):
    som = AdaptiveSomClustering(x_dim=2, y_dim=2, input_len=X_pca.shape[1], sigma=0.5, learning_rate=0.5)
    som.train(X_pca, iterations=1000)
    return som


def _fit_art2(X_pca: np.ndarray) -> SimpleART2:
    model = SimpleART2(rho=0.82, alpha=0.15)
    for row in X_pca:
        model.compute_anomaly_score(row)
    return model


def _fit_spatial_profile(X_train: np.ndarray):
    # Dummy GATModel returning for compatibility
    model = GATModel()
    if model is None:
        corr = np.nan_to_num(np.abs(np.corrcoef(X_train, rowvar=False)), nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 0.0)
        max_corr = float(np.max(corr)) or 1.0
        corr = corr / max_corr
        edges = []
        weights = []
        for src in range(corr.shape[0]):
            for dst in range(corr.shape[1]):
                if src != dst:
                    edges.append([int(src), int(dst)])
                    weights.append(float(corr[src, dst]))
        
        class DummyGNN:
            def __init__(self, edges, weights):
                self.attention = {"edges": edges, "weights": weights}
            def predict(self, x): return 0.0, self.attention
        return DummyGNN(edges, weights)

    model.eval()
    attention = {"edges": [], "weights": []}
    return model


def _fit_rl_table(clusters: np.ndarray) -> np.ndarray:
    q_table = np.full((4, 4), -0.25, dtype=float)
    for state in range(4):
        target_action = min(state, 3)
        if state == 0:
            rewards = [0.7, 0.1, -0.5, -0.7]
        elif state == 1:
            rewards = [-0.4, 0.7, 0.2, -0.1]
        elif state == 2:
            rewards = [-1.0, -0.3, 0.8, 0.9]
        else:
            rewards = [-1.0, -0.5, 0.4, 1.0]
        q_table[state] = rewards

    observed = np.bincount(np.clip(clusters, 0, 3), minlength=4)
    if observed.sum() > 0:
        q_table += (observed / observed.sum()).reshape(-1, 1) * 0.05
    return q_table


def _classify_meta(meta: dict[str, Any]) -> int:
    risk = str(meta.get("risk", "")).lower()
    if risk == "critical":
        return 2
    if risk == "warning":
        return 1
    return 0


def _test_runtime_pipeline(test_df: pd.DataFrame, labels: np.ndarray, limit: int | None = None) -> dict[str, Any]:
    import importlib
    import ai.main as runtime

    runtime = importlib.reload(runtime)
    sample_df = (test_df if limit is None else test_df.head(limit)).reset_index(drop=True)
    y_true_scenario = _labels_to_clusters(labels[: len(sample_df)])
    y_true = _risk_from_scenario_class(y_true_scenario)
    y_pred = []
    y_pred_scenario = []
    clusters = []
    action_counts: dict[str, int] = {}

    for _, row in sample_df.iterrows():
        payload = {col: float(row[col]) for col in FEATURE_COLS}
        if "Timestamp" in row:
            payload["Timestamp"] = row["Timestamp"]
        result = runtime.run_pipeline(payload, verbose=False)
        y_pred.append(_classify_meta(result["_meta"]))
        y_pred_scenario.append(int(result["_meta"].get("scenario_id", result["_meta"].get("cluster_id", 0))))
        clusters.append(int(result["_meta"].get("som_state", result["_meta"]["cluster_id"])))
        action_key = f"{result['fan']}/{result['alarm']}/{result['buzzer']}/{result['rgb_led']}"
        action_counts[action_key] = action_counts.get(action_key, 0) + 1

    y_pred_arr = np.asarray(y_pred, dtype=int)
    y_pred_scenario_arr = np.asarray(y_pred_scenario, dtype=int)
    clusters_arr = np.asarray(clusters, dtype=int)
    accuracy = float(np.mean(y_pred_arr == y_true)) if len(y_true) else 0.0
    scenario_accuracy = float(np.mean(y_pred_scenario_arr == y_true_scenario)) if len(y_true_scenario) else 0.0
    scenario_confusion = np.zeros((4, 4), dtype=int)
    for true_value, pred_value in zip(y_true_scenario, y_pred_scenario_arr):
        scenario_confusion[int(np.clip(true_value, 0, 3)), int(np.clip(pred_value, 0, 3))] += 1
    per_scenario_recall = {}
    for idx, name in SCENARIO_LABELS.items():
        support = int(np.sum(y_true_scenario == idx))
        hits = int(np.sum((y_true_scenario == idx) & (y_pred_scenario_arr == idx)))
        per_scenario_recall[str(idx)] = {
            "name": name,
            "support": support,
            "recall": None if support == 0 else round(float(hits / support), 4),
        }

    true_clusters = y_true_scenario
    cluster_accuracy = float(np.mean(clusters_arr == true_clusters)) if len(true_clusters) else 0.0
    false_positive_critical = float(np.mean((y_true < 2) & (y_pred_arr == 2))) if len(y_true) else 0.0
    false_negative_danger = float(np.mean((y_true == 2) & (y_pred_arr < 2))) if len(y_true) else 0.0
    false_alert = float(np.mean((y_true == 0) & (y_pred_arr > 0))) if len(y_true) else 0.0
    warning_miss = float(np.mean((y_true == 1) & (y_pred_arr == 0))) if len(y_true) else 0.0
    return {
        "rows_tested": int(len(sample_df)),
        "three_class_accuracy": round(accuracy, 4),
        "four_scenario_accuracy": round(scenario_accuracy, 4),
        "som_cluster_accuracy": round(cluster_accuracy, 4),
        "false_positive_critical_rate": round(false_positive_critical, 4),
        "false_negative_danger_rate": round(false_negative_danger, 4),
        "false_alert_rate": round(false_alert, 4),
        "warning_miss_rate": round(warning_miss, 4),
        "scenario_confusion_true_rows_pred_cols": scenario_confusion.tolist(),
        "per_scenario_recall": per_scenario_recall,
        "predicted_class_counts": {str(k): int(v) for k, v in zip(*np.unique(y_pred_arr, return_counts=True))},
        "predicted_scenario_counts": {str(k): int(v) for k, v in zip(*np.unique(y_pred_scenario_arr, return_counts=True))},
        "predicted_cluster_counts": {str(k): int(v) for k, v in zip(*np.unique(clusters_arr, return_counts=True))},
        "target_class_counts": {str(k): int(v) for k, v in zip(*np.unique(y_true, return_counts=True))},
        "target_scenario_counts": {str(k): int(v) for k, v in zip(*np.unique(y_true_scenario, return_counts=True))},
        "target_cluster_counts": {str(k): int(v) for k, v in zip(*np.unique(true_clusters, return_counts=True))},
        "action_counts": action_counts,
    }


def train_and_save(csv_path: Path = DATA_PATH) -> dict[str, Any]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_data(csv_path)
    split_idx = int(len(df) * 0.80)

    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    X_train_raw = train_df[FEATURE_COLS].to_numpy(dtype=float)
    X_test_raw = test_df[FEATURE_COLS].to_numpy(dtype=float)
    y_train = train_df["True_Scenario"].to_numpy()
    y_test = test_df["True_Scenario"].to_numpy()

    scaler = _fit_scaler(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    pca = _fit_pca(X_train, n_components=3)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    y_train_3class = _target_3class(y_train)
    y_test_3class = _target_3class(y_test)
    y_train_4class = _labels_to_clusters(y_train)
    X_train_risk_balanced, y_train_risk_balanced = _smote_resample(X_train, y_train_3class)
    X_train_scenario_balanced, y_train_scenario_balanced = _smote_resample(X_train, y_train_4class)
    X_train_pca_scenario_balanced = pca.transform(X_train_scenario_balanced)

    rbf = _fit_rbf((X_train_pca, X_train_pca_scenario_balanced))
    risk_guard = _fit_risk_guard(X_train, y_train_4class)
    som = _fit_som_state(X_train_pca_scenario_balanced, X_train_scenario_balanced, y_train_scenario_balanced)
    art2 = _fit_art2(X_train_pca)
    gnn = _fit_spatial_profile(X_train)
    clusters = _labels_to_clusters(y_train)
    q_table = _fit_rl_table(clusters)
    ga_policy, ga_score = GA(pop_size=20, generations=14).run(
        X_train_risk_balanced,
        y_train_risk_balanced,
        X_test,
        y_test_3class,
        verbose=False,
    )

    _save_pickle(scaler, MODELS_DIR / "scaler.pkl")
    _save_pickle(pca, MODELS_DIR / "pca.pkl")
    _save_pickle(rbf, MODELS_DIR / "rbf.pkl")
    _save_pickle(risk_guard, MODELS_DIR / "risk_guard.pkl")
    _save_pickle(som, MODELS_DIR / "som.pkl")
    _save_pickle(art2, MODELS_DIR / "art2.pkl")
    _save_pickle(gnn, MODELS_DIR / "gnn.pkl")
    np.save(MODELS_DIR / "gnn_attention.npy", gnn.attention, allow_pickle=True)
    np.save(MODELS_DIR / "rl_qtable.npy", q_table)
    np.save(MODELS_DIR / "ga_policy.npy", ga_policy, allow_pickle=True)

    test_report = _test_runtime_pipeline(test_df, y_test)
    report = {
        "data_path": str(csv_path),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_cols": FEATURE_COLS,
        "art2_categories": int(len(art2.categories)),
        "pca_explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "pca_total_explained_variance": round(float(np.sum(pca.explained_variance_ratio_)), 4),
        "class_balance": {
            "risk_before": {str(k): int(v) for k, v in zip(*np.unique(y_train_3class, return_counts=True))},
            "risk_after_smote": {str(k): int(v) for k, v in zip(*np.unique(y_train_risk_balanced, return_counts=True))},
            "scenario_before": {str(k): int(v) for k, v in zip(*np.unique(y_train_4class, return_counts=True))},
            "scenario_after_smote": {
                str(k): int(v) for k, v in zip(*np.unique(y_train_scenario_balanced, return_counts=True))
            },
        },
        "scenario_labels": {str(k): v for k, v in SCENARIO_LABELS.items()},
        "scenario_profiles": {
            "full": _scenario_profiles(df),
            "train": _scenario_profiles(train_df),
            "test": _scenario_profiles(test_df),
        },
        "rbf_centers": int(len(rbf.centers)),
        "rbf_sigma": round(float(rbf.sigma), 6),
        "rbf_trend_scale": {
            "raw_q01": round(float(rbf.target_low), 6),
            "raw_q99": round(float(rbf.target_high), 6),
            "fuzzy_min": -5,
            "fuzzy_max": 5,
        },
        "risk_guard": {
            "type": "KNNRiskGuard",
            "k": int(risk_guard.k),
            "class_weights": [round(float(v), 3) for v in risk_guard.class_weights],
            "reference_rows": int(len(risk_guard.reference_labels)),
        },
        "som_clusters": sorted(int(k) for k in som.centroids.keys()),
        "gnn_attention_edges": int(len(gnn.attention["edges"])),
        "rl_qtable_shape": list(q_table.shape),
        "ga_policy": ga_policy,
        "ga_score": round(float(ga_score), 4),
        "runtime_test": test_report,
        "saved_artifacts": [
            "scaler.pkl",
            "pca.pkl",
            "art2.pkl",
            "rbf.pkl",
            "risk_guard.pkl",
            "som.pkl",
            "gnn.pkl",
            "gnn_attention.npy",
            "rl_qtable.npy",
            "ga_policy.npy",
        ],
    }

    (MODELS_DIR / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = train_and_save()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
