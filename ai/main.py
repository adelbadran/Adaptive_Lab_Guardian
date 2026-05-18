"""Runtime pipeline for the Adaptive Lab Guardian project.

Diagram fit:
Sensor data -> PCA noise filter -> ART/RBF/GNN/SOM intelligence fan-out
-> fuzzy baseline decision -> RL refinement -> final actuator action.
"""

from __future__ import annotations

import importlib
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

AI_DIR = Path(__file__).resolve().parent
MODELS_DIR = AI_DIR / "models"

FEATURE_COLS = ["Temp_C", "Humidity_pct", "Gas_AQI", "Light_Lux", "Motion_Detected"]
SENSOR_NAMES = ["Temp", "Humidity", "Gas", "Light", "Motion"]

SOM_CLUSTER_LABELS = {
    0: "Normal",
    1: "Crowded",
    2: "Chemical",
    3: "Security",
}

DASHBOARD_STATE_LABELS = {
    0: "Normal",
    1: "Warning",
    2: "Dangerous",
    3: "Dangerous",
}

RISK_STATE = {
    "Safe": (0, "Normal"),
    "Warning": (1, "Warning"),
    "Critical": (2, "Dangerous"),
}

DEFAULT_ACTION = {
    "fan": "OFF",
    "alarm": "OFF",
    "servo": "CLOSED",
    "buzzer": "OFF",
    "rgb_led": "GREEN",
}

FALLBACK_RANGES = np.array([50.0, 100.0, 1000.0, 1000.0, 1.0], dtype=float)
FALLBACK_TREND_RAW_LIMIT = 0.34

_last_filtered: np.ndarray | None = None


def _optional_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


torch = _optional_import("torch")
gnn_mod = _optional_import("ai.gnn")
pca_mod = _optional_import("ai.pca")
rbf_mod = _optional_import("ai.rbf")
som_mod = _optional_import("ai.som")
art2_mod = _optional_import("ai.art2")
fuzzy_mod = _optional_import("ai.fuzzy")
rl_mod = _optional_import("ai.rl")


def _load_numpy(filename: str):
    path = MODELS_DIR / filename
    if not path.exists():
        return None

    arr = np.load(path, allow_pickle=True)
    try:
        return arr.item()
    except Exception:
        return arr


def _load_model(filename: str, model_class: Any | None = None):
    path = MODELS_DIR / filename
    if not path.exists():
        return None

    try:
        if filename.endswith(".pth"):
            if torch is None:
                return None
            state = torch.load(str(path), map_location="cpu")
            if model_class is not None and isinstance(state, dict):
                model = model_class()
                model.load_state_dict(state)
                model.eval()
                return model
            return state

        try:
            import joblib

            return joblib.load(path)
        except Exception:
            with path.open("rb") as f:
                return pickle.load(f)
    except Exception:
        return None


def _get_gnn_class():
    if gnn_mod is None:
        return None
    return getattr(gnn_mod, "GATModel", None)


gnn_model = _load_model("gnn.pth", model_class=_get_gnn_class())
gnn_profile = _load_model("gnn.pkl")
pca_model = _load_model("pca.pkl")
rbf_model = _load_model("rbf.pkl")
risk_guard_model = _load_model("risk_guard.pkl")
som_model = _load_model("som.pkl")
art2_model = _load_model("art2.pkl")
scaler_model = _load_model("scaler.pkl")
rl_table = _load_numpy("rl_qtable.npy")
ga_policy = _load_numpy("ga_policy.npy")
saved_attention = _load_numpy("gnn_attention.npy")

if rl_mod is not None and hasattr(rl_mod, "configure"):
    rl_mod.configure(q_table=rl_table)


def _as_flat_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def _clip01(value: float | np.ndarray) -> float | np.ndarray:
    return np.clip(value, 0.0, 1.0)


def _scale_trend_for_fuzzy(raw_trend: float, model: Any | None = None) -> float:
    """Convert RBF/raw trend values into the fuzzy controller range [-5, 5]."""
    raw = float(np.asarray(raw_trend, dtype=float).reshape(-1)[0])

    if model is not None and hasattr(model, "scale_trend"):
        try:
            return float(np.asarray(model.scale_trend(raw)).reshape(-1)[0])
        except Exception:
            pass

    if model is not None and hasattr(model, "scaler"):
        return float(np.clip(raw, -5.0, 5.0))

    return float(np.clip((raw / FALLBACK_TREND_RAW_LIMIT) * 5.0, -5.0, 5.0))


def _sanitize_action(action: dict[str, Any]) -> dict[str, str]:
    clean = DEFAULT_ACTION.copy()
    for key in clean:
        if key in action:
            clean[key] = str(action[key]).upper()
    if clean["rgb_led"] not in {"GREEN", "YELLOW", "RED"}:
        clean["rgb_led"] = "GREEN"
    return clean


def _create_edge_index(num_nodes: int):
    if torch is None:
        return None

    edges = [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j]
    return torch.tensor(edges, dtype=torch.long).t()


def _static_attention(num_nodes: int) -> dict[str, Any]:
    if isinstance(saved_attention, dict):
        return saved_attention

    weights = np.ones((num_nodes, num_nodes), dtype=float) - np.eye(num_nodes, dtype=float)
    edges = []
    edge_weights = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edges.append([i, j])
                edge_weights.append(float(weights[i, j] / max(num_nodes - 1, 1)))
    return {"edges": edges, "weights": edge_weights}


def step_pca(x: np.ndarray) -> np.ndarray:
    """PCA noise filter from the input stage."""
    vector = _as_flat_array(x)

    for candidate in (pca_model, pca_mod):
        if candidate is not None and hasattr(candidate, "transform"):
            try:
                return _as_flat_array(candidate.transform(vector.reshape(1, -1)))
            except Exception:
                continue

    return vector


def step_art2(x_filtered: np.ndarray) -> tuple[bool, float]:
    """ART anomaly detection over the PCA-filtered representation."""
    vector = _as_flat_array(x_filtered)

    for candidate in (art2_model, art2_mod):
        if candidate is None:
            continue
        if hasattr(candidate, "compute_anomaly_score"):
            try:
                score = float(candidate.compute_anomaly_score(vector))
                return score >= 0.35, float(_clip01(score))
            except Exception:
                pass
        if hasattr(candidate, "is_new_pattern"):
            try:
                is_new = bool(candidate.is_new_pattern(vector))
                return is_new, 1.0 if is_new else 0.0
            except Exception:
                pass

    score = float(max(np.max(vector) - 0.92, 0.02 - np.min(vector), 0.0) / 0.08)
    score = float(_clip01(score))
    return score >= 0.5, score


def step_rbf(x_filtered: np.ndarray) -> dict[str, float]:
    """RBF trend signal over the PCA-filtered representation."""
    global _last_filtered

    vector = _as_flat_array(x_filtered)
    raw_trend = 0.0

    if rbf_mod is not None and hasattr(rbf_mod, "step_rbf"):
        try:
            out = rbf_mod.step_rbf(vector, model=rbf_model)
            if isinstance(out, dict) and "trend" in out:
                raw_trend = float(out["trend"])
                return {
                    "trend": _scale_trend_for_fuzzy(raw_trend, model=rbf_model),
                    "raw_trend": raw_trend,
                }
        except Exception:
            pass

    if rbf_model is not None and hasattr(rbf_model, "predict"):
        try:
            pred = _as_flat_array(rbf_model.predict(vector.reshape(1, -1)))
            raw_trend = float(pred[0])
            return {
                "trend": _scale_trend_for_fuzzy(raw_trend, model=rbf_model),
                "raw_trend": raw_trend,
            }
        except Exception:
            pass

    if _last_filtered is None or _last_filtered.shape != vector.shape:
        raw_trend = 0.0
    else:
        raw_trend = float(np.mean(vector - _last_filtered))
    _last_filtered = vector.copy()
    return {"trend": _scale_trend_for_fuzzy(raw_trend), "raw_trend": raw_trend}


def step_risk_guard(x_scaled: np.ndarray) -> dict[str, Any]:
    """Data-driven guardrail trained from the historical CSV labels."""
    if risk_guard_model is not None and hasattr(risk_guard_model, "predict"):
        try:
            out = risk_guard_model.predict(_as_flat_array(x_scaled))
            return {
                "risk_class": int(out.get("risk_class", 0)),
                "scenario_class": int(out.get("scenario_class", out.get("risk_class", 0))),
                "scenario_label": str(out.get("scenario_label", "Unknown")),
                "confidence": float(out.get("confidence", 0.0)),
                "margin": float(out.get("margin", 0.0)),
                "votes": [float(v) for v in out.get("votes", [0.0, 0.0, 0.0, 0.0])],
                "risk_votes": [float(v) for v in out.get("risk_votes", [0.0, 0.0, 0.0])],
            }
        except Exception:
            pass

    return {
        "risk_class": 0,
        "scenario_class": 0,
        "scenario_label": "Normal",
        "confidence": 0.0,
        "margin": 0.0,
        "votes": [0.0, 0.0, 0.0, 0.0],
        "risk_votes": [0.0, 0.0, 0.0],
    }


def step_gnn(x_filtered: np.ndarray) -> tuple[float, dict[str, Any]]:
    """GNN spatial relationship signal and attention metadata."""
    vector = _as_flat_array(x_filtered)
    attention = _static_attention(len(vector))

    if gnn_profile is not None:
        try:
            if hasattr(gnn_profile, "predict"):
                out = gnn_profile.predict(vector)
                if isinstance(out, tuple):
                    return float(_clip01(out[0])), out[1]
                return float(_clip01(out)), attention
            if hasattr(gnn_profile, "spatial_risk"):
                return float(_clip01(gnn_profile.spatial_risk(vector))), attention
        except Exception:
            pass

    if gnn_model is not None and torch is not None and len(vector) == 5:
        try:
            edge_index = _create_edge_index(len(vector))
            x_tensor = torch.tensor(vector, dtype=torch.float).view(len(vector), 1)
            batch = torch.zeros(len(vector), dtype=torch.long)
            with torch.no_grad():
                emb, attn1, _ = gnn_model(x_tensor, edge_index, batch, return_attention=True)
            edge_idx, weights = attn1
            weights = weights.mean(dim=1)
            attention = {
                "edges": edge_idx.t().cpu().numpy().tolist(),
                "weights": weights.cpu().numpy().tolist(),
            }
            spatial_risk = float(_clip01(np.mean(np.abs(emb.cpu().numpy()))))
            return spatial_risk, attention
        except Exception:
            pass

    if gnn_mod is not None and hasattr(gnn_mod, "spatial_risk"):
        try:
            return float(_clip01(gnn_mod.spatial_risk(vector))), attention
        except Exception:
            pass

    spatial_risk = float(_clip01(np.std(vector) + np.mean(vector) * 0.25))
    return spatial_risk, attention


def step_som(x_filtered: np.ndarray, x_scaled: np.ndarray) -> int:
    """SOM state cluster from PCA-filtered features."""
    vector = _as_flat_array(x_filtered)
    scaled = _as_flat_array(x_scaled)

    if som_model is not None and hasattr(som_model, "predict_cluster_from_scaled"):
        try:
            pred = som_model.predict_cluster_from_scaled(scaled)
            return int(np.asarray(pred).reshape(-1)[0])
        except Exception:
            pass

    for method_name in ("predict_cluster", "predict"):
        if som_model is not None and hasattr(som_model, method_name):
            try:
                pred = getattr(som_model, method_name)(vector)
                return int(np.asarray(pred).reshape(-1)[0])
            except Exception:
                pass
        if som_mod is not None and hasattr(som_mod, method_name):
            try:
                pred = getattr(som_mod, method_name)(vector)
                return int(np.asarray(pred).reshape(-1)[0])
            except Exception:
                pass

    temp, gas, light, motion = scaled[0], scaled[2], scaled[3], scaled[4]
    
    # BREACH: lenient with other readings. Just motion + low light
    if motion >= 0.5 and light < 0.35:
        return 3
    if gas > 0.62 or temp > 0.72:
        return 2
    if gas > 0.35 or temp > 0.55 or motion >= 0.5:
        return 1
    return 0


def step_fuzzy(context: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    """Fuzzy baseline decision before the RL refinement stage."""
    if fuzzy_mod is not None and hasattr(fuzzy_mod, "decide"):
        try:
            result = fuzzy_mod.decide(**context)
            decision_meta = dict(result.pop("_decision", {})) if isinstance(result, dict) else {}
            return _sanitize_action(result), decision_meta
        except Exception:
            pass

    # GNN's spatial_risk weight increased to 25.0 so it has a major impact on the final risk score
    risk_score = (
        100.0 * context["anomaly_score"] * 0.35
        + max(context["trend"], 0.0) * 12.0
        + context["cluster_id"] * 18.0
        + context["gas_level"] * 30.0
        + context["temp_level"] * 20.0
        + context["spatial_risk"] * 25.0  # Increased GNN Impact
        + context.get("guard_risk_class", 0) * context.get("guard_confidence", 0.0) * 18.0
    )
    risk_score = float(np.clip(risk_score, 0.0, 100.0))

    security_signature = (
        context["cluster_id"] == 3
        or (context.get("motion", 0.0) >= 0.5 and context.get("light_level", 1.0) < 0.35)
    )

    # GA Policy Integration: The Genetic Algorithm optimizes these thresholds based on environment evolution
    policy = context.get("threshold_policy")
    if not isinstance(policy, dict):
        policy = {"critical": 70, "warning": 40, "safe_max": 50}
    
    crit_thresh = policy.get("critical", 70)
    warn_thresh = policy.get("warning", 40)
    safe_max_thresh = policy.get("safe_max", 50)

    if security_signature:
        action = {"fan": "OFF", "alarm": "ON", "servo": "CLOSED", "buzzer": "ON", "rgb_led": "RED"}
        risk = "Critical"
        scenario = "Breach"
    elif risk_score >= crit_thresh or context.get("guard_risk_class", 0) >= 2 or context["cluster_id"] == 2:
        action = {"fan": "ON", "alarm": "ON", "servo": "OPEN", "buzzer": "ON", "rgb_led": "RED"}
        risk = "Critical"
        scenario = "Chemical"
    elif risk_score >= warn_thresh or context.get("guard_risk_class", 0) >= 1 or context["cluster_id"] == 1:
        if context["cluster_id"] == 0 and risk_score < safe_max_thresh and context.get("guard_risk_class", 0) == 0:
            action = DEFAULT_ACTION.copy()
            risk = "Safe"
            scenario = "Normal"
        else:
            action = {"fan": "ON", "alarm": "OFF", "servo": "CLOSED", "buzzer": "OFF", "rgb_led": "YELLOW"}
            risk = "Warning"
            scenario = "Crowded"
    else:
        action = DEFAULT_ACTION.copy()
        risk = "Safe"
        scenario = "Normal"

    return action, {
        "scenario": scenario,
        "risk": risk,
        "risk_score": risk_score,
        "security_signature": security_signature,
    }


def step_rl(cluster_id: int, action: dict[str, str], reward: float, context: dict[str, Any]) -> dict[str, str]:
    """RL agent refinement after fuzzy baseline."""
    refined_action = _sanitize_action(action)
    
    # 1. External RL Module (if loaded)
    if rl_mod is not None and hasattr(rl_mod, "update"):
        try:
            return _sanitize_action(rl_mod.update(cluster_id, action, reward, context=context))
        except TypeError:
            try:
                return _sanitize_action(rl_mod.update(cluster_id, action, reward))
            except Exception:
                pass
        except Exception:
            pass

    # 2. RL Learned Policy Application (Fallback/Embedded Q-Table)
    # RL learns from the 'reward' signal over time. If the Q-Table exists, we apply its overrides.
    if isinstance(rl_table, dict):
        state_key = f"{cluster_id}_{context.get('risk', 'Safe')}"
        if state_key in rl_table:
            override = rl_table[state_key]
            for k, v in override.items():
                if k in refined_action:
                    refined_action[k] = str(v).upper()

    # If the system is Safe, but temperature is slightly elevated, the RL agent 
    # might learn to turn on the fan proactively to maximize the long-term reward.
    # However, to maintain strict 'Normal' behavior, we leave the action as is.
    if context.get("risk") == "Safe" and context.get("temp_level", 0.0) > 0.55:
        pass # Action left as DEFAULT_ACTION to ensure "Normal" behaves normally
        
    return refined_action


def compute_reward(cluster_id: int, action: dict[str, str], context: dict[str, Any]) -> float:
    """Reward signal for the RL and evolution stages."""
    risk_score = float(context.get("risk_score", 0.0))
    risk = str(context.get("risk", "Safe"))
    is_anomaly = bool(context.get("is_anomaly", False))
    warning_action = action["fan"] == "ON" or action["servo"] == "OPEN" or action["rgb_led"] == "YELLOW"
    critical_action = action["alarm"] == "ON" and action["fan"] == "ON" and action["rgb_led"] == "RED"
    security_action = action["alarm"] == "ON" and action["buzzer"] == "ON" and action["rgb_led"] == "RED"

    if risk == "Critical":
        if critical_action or security_action:
            return 1.0
        return -1.0

    if risk == "Warning":
        if warning_action and action["buzzer"] == "OFF":
            return 0.7
        return -0.4

    if is_anomaly and action["buzzer"] != "ON" and risk_score >= 70:
        return -0.75

    if action["alarm"] == "OFF" and action["buzzer"] == "OFF" and action["fan"] == "OFF":
        return 0.5
    return -0.2


def action_id_from_decision(action: dict[str, str], meta: dict[str, Any]) -> int:
    """Map the Python action contract to the ESP32 mode IDs."""
    risk = str(meta.get("risk", "Safe"))
    scenario = str(meta.get("scenario", ""))
    cluster_id = int(meta.get("cluster_id", 0))
    scenario_id = int(meta.get("scenario_id", cluster_id))

    if risk == "Critical":
        if scenario_id == 3 or cluster_id == 3 or scenario in {"Breach", "Security"} or bool(meta.get("security_signature", False)):
            return 3
        if scenario_id == 2 or scenario in {"Chemical", "Hazardous"}:
            return 2
        return 2
    if risk == "Warning":
        return 1

    alarm_on = str(action.get("alarm", "OFF")).upper() == "ON"
    buzzer_on = str(action.get("buzzer", "OFF")).upper() == "ON"
    fan_on = str(action.get("fan", "OFF")).upper() == "ON"
    if alarm_on or buzzer_on:
        return 2
    if fan_on:
        return 1
    return 0


def validate_sensor_data(sensor_data: dict[str, Any]) -> np.ndarray:
    missing = [name for name in FEATURE_COLS if name not in sensor_data]
    if missing:
        raise ValueError(f"Missing sensor keys: {missing}")
    return np.array([sensor_data[name] for name in FEATURE_COLS], dtype=float)


def scale_sensor_data(raw: np.ndarray, scaler=None) -> np.ndarray:
    selected_scaler = scaler if scaler is not None else scaler_model
    if selected_scaler is not None and hasattr(selected_scaler, "transform"):
        return _as_flat_array(selected_scaler.transform(raw.reshape(1, -1)))
    return _clip01(raw / FALLBACK_RANGES)


def run_pipeline(sensor_data: dict[str, Any], scaler=None, verbose: bool = True) -> dict[str, Any]:
    """Run one real-time sensor reading through the project architecture."""

    def log(step: str, message: str):
        if verbose:
            print(f"[{step}] {message}")

    raw = validate_sensor_data(sensor_data)
    x_scaled = scale_sensor_data(raw, scaler=scaler)
    log("01.INPUT", f"scaled={np.round(x_scaled, 4).tolist()}")

    x_filtered = step_pca(x_scaled)
    log("01.PCA", f"filtered_shape={x_filtered.shape} values={np.round(x_filtered, 4).tolist()}")

    is_anomaly, anomaly_score = step_art2(x_filtered)
    rbf_out = step_rbf(x_filtered)
    trend = float(rbf_out["trend"])
    raw_trend = float(rbf_out.get("raw_trend", trend))
    spatial_risk, attention = step_gnn(x_filtered)
    som_cluster_id = int(np.clip(step_som(x_filtered, x_scaled), 0, 3))
    guard = step_risk_guard(x_scaled)
    decision_cluster_id = int(guard["scenario_class"]) if float(guard["confidence"]) >= 0.34 else som_cluster_id

    context = {
        "anomaly_score": anomaly_score,
        "is_anomaly": is_anomaly,
        "trend": trend,
        "raw_trend": raw_trend,
        "cluster_id": decision_cluster_id,
        "som_cluster_id": som_cluster_id,
        "spatial_risk": spatial_risk,
        "motion": float(x_scaled[4]),
        "humidity_level": float(x_scaled[1]),
        "gas_level": float(x_scaled[2]),
        "light_level": float(x_scaled[3]),
        "temp_level": float(x_scaled[0]),
        "guard_risk_class": int(guard["risk_class"]),
        "guard_scenario_class": int(guard["scenario_class"]),
        "guard_confidence": float(guard["confidence"]),
        "guard_margin": float(guard["margin"]),
        "threshold_policy": ga_policy if isinstance(ga_policy, dict) else None,
    }
    log(
        "02.INTEL",
        " ".join(
            [
                f"anomaly={anomaly_score:.3f}",
                f"trend_raw={raw_trend:.4f}",
                f"trend_fuzzy={trend:.3f}",
                f"spatial={spatial_risk:.3f}",
                f"som={som_cluster_id}",
                f"decision_cluster={decision_cluster_id}",
                f"guard={guard['scenario_label']}({guard['risk_class']})@{guard['confidence']:.2f}",
            ]
        ),
    )

    baseline_action, decision_meta = step_fuzzy(context)
    context["risk"] = str(decision_meta.get("risk", "Safe"))
    context["risk_score"] = float(decision_meta.get("risk_score", 0.0))
    reward = compute_reward(decision_cluster_id, baseline_action, context)
    final_action = step_rl(decision_cluster_id, baseline_action, reward, context)
    log("03.DECIDE", f"baseline={baseline_action} reward={reward:+.2f} final={final_action}")

    cluster_id = decision_cluster_id
    cluster_label = SOM_CLUSTER_LABELS.get(cluster_id, "Unknown")
    som_cluster_label = SOM_CLUSTER_LABELS.get(som_cluster_id, "Unknown")
    default_state = (min(cluster_id, 2), DASHBOARD_STATE_LABELS.get(cluster_id, "Unknown"))
    state, state_label = RISK_STATE.get(context["risk"], default_state)
    scenario_id = int(np.clip(decision_meta.get("scenario_id", guard["scenario_class"]), 0, 3))
    scenario_label = SOM_CLUSTER_LABELS.get(scenario_id, str(decision_meta.get("scenario", cluster_label)))

    meta = {
        "state": state,
        "state_label": state_label,
        "cluster_id": cluster_id,
        "cluster_label": cluster_label,
        "som_state": som_cluster_id,
        "som_state_label": som_cluster_label,
        "scenario_id": scenario_id,
        "scenario": scenario_label,
        "fuzzy_scenario": decision_meta.get("scenario", cluster_label),
        "risk": decision_meta.get("risk", "Unknown"),
        "risk_score": round(float(decision_meta.get("risk_score", 0.0)), 2),
        "security_signature": bool(decision_meta.get("security_signature", False)),
        "gas_pred": round(float(x_scaled[2]), 4),
        "temp_pred": round(float(x_scaled[0]), 4),
        "humidity_level": round(float(x_scaled[1]), 4),
        "light_level": round(float(x_scaled[3]), 4),
        "trend": round(trend, 4),
        "raw_trend": round(raw_trend, 6),
        "trend_scale": [-5, 5],
        "spatial_risk": round(spatial_risk, 4),
        "anomaly_score": round(anomaly_score, 4),
        "is_anomaly": bool(is_anomaly),
        "guard_risk_class": int(guard["risk_class"]),
        "guard_scenario_class": int(guard["scenario_class"]),
        "guard_scenario_label": str(guard["scenario_label"]),
        "guard_confidence": round(float(guard["confidence"]), 4),
        "guard_margin": round(float(guard["margin"]), 4),
        "guard_votes": [round(float(v), 3) for v in guard["votes"]],
        "guard_risk_votes": [round(float(v), 3) for v in guard["risk_votes"]],
        "reward": round(float(reward), 4),
        "baseline_action": baseline_action,
        "attention": attention,
        "pca_features": np.round(x_filtered, 6).tolist(),
        "evolution": {
            "environment": "Adaptive Lab Guardian",
            "reward": round(float(reward), 4),
            "ga_update_ready": True,
        },
    }
    action_id = action_id_from_decision(final_action, meta)
    meta["action_id"] = action_id

    result = {
        **final_action,
        "action_id": action_id,
        "_meta": meta,
    }

    return result


def format_pipeline_output(result: dict[str, Any], sensor_data: dict[str, Any] | None = None) -> str:
    """Human-readable runtime report for python -m ai.main and MQTT logs."""
    meta = result.get("_meta", {})
    lines = ["Adaptive Lab Guardian Decision", "-" * 64]

    if sensor_data:
        lines.extend(
            [
                "Sensors",
                f"  Temp={float(sensor_data['Temp_C']):6.2f} C   Humidity={float(sensor_data['Humidity_pct']):6.2f} %",
                f"  Gas ={float(sensor_data['Gas_AQI']):6.2f} AQI Light={float(sensor_data['Light_Lux']):8.2f} Lux Motion={int(sensor_data['Motion_Detected'])}",
            ]
        )

    lines.extend(
        [
            "Decision",
            f"  State={meta.get('state_label')} Risk={meta.get('risk')} Score={meta.get('risk_score')} Scenario={meta.get('scenario_id')}:{meta.get('scenario')}",
            f"  Action ID={meta.get('action_id')} Cluster={meta.get('cluster_id')} Guard={meta.get('guard_scenario_label')}@{meta.get('guard_confidence')}",
            "Intelligence",
            f"  Anomaly={meta.get('anomaly_score')} Spatial={meta.get('spatial_risk')} Trend(raw={meta.get('raw_trend')}, fuzzy={meta.get('trend')})",
            f"  GasScaled={meta.get('gas_pred')} TempScaled={meta.get('temp_pred')} LightScaled={meta.get('light_level')}",
            "Actuators",
            f"  Fan={result.get('fan')} Alarm={result.get('alarm')} Servo={result.get('servo')} Buzzer={result.get('buzzer')} RGB={result.get('rgb_led')}",
            "-" * 64,
        ]
    )
    return "\n".join(lines)


def run_batch(df, scaler=None, verbose: bool = False) -> list[dict[str, Any]]:
    results = []
    for _, row in df.iterrows():
        sensor_data = {col: row[col] for col in FEATURE_COLS if col in row}
        results.append(run_pipeline(sensor_data, scaler=scaler, verbose=verbose))
    return results


if __name__ == "__main__":
    test_cases = [
        {
            "Temp_C": 27,
            "Humidity_pct": 55,
            "Gas_AQI": 77,
            "Light_Lux": 0,
            "Motion_Detected": 0,
        },
        {
            "Temp_C": 24.5,
            "Humidity_pct": 62.0,
            "Gas_AQI": 70.0,
            "Light_Lux": 48062.0,
            "Motion_Detected": 0,
        },
        {
            "Temp_C": 30.0,
            "Humidity_pct": 61.0,
            "Gas_AQI": 200,
            "Light_Lux": 60904.0,
            "Motion_Detected": 1,
        },
        {
            "Temp_C": 25.0,
            "Humidity_pct": 61.0,
            "Gas_AQI": 100,
            "Light_Lux": 0,
            "Motion_Detected": 1,
        }
    ]

    for case in test_cases:
        output = run_pipeline(case, verbose=False)
        print(format_pipeline_output(output, case))
