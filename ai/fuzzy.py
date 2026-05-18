"""Fuzzy baseline decision engine for Adaptive Lab Guardian.

The module accepts the four intelligence signals shown in the architecture:
ART anomaly, RBF trend, GNN spatial risk, and SOM cluster. If scikit-fuzzy is
available it uses a fuzzy controller; otherwise it falls back to deterministic
membership-inspired rules with the same public API.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
except Exception:  # pragma: no cover - optional dependency
    fuzz = None
    ctrl = None


ACTION_SAFE = {"fan": "OFF", "alarm": "OFF", "servo": "CLOSED", "buzzer": "OFF", "rgb_led": "GREEN"}
ACTION_WARNING = {"fan": "ON", "alarm": "OFF", "servo": "CLOSED", "buzzer": "OFF", "rgb_led": "YELLOW"}
ACTION_CRITICAL = {"fan": "ON", "alarm": "ON", "servo": "OPEN", "buzzer": "ON", "rgb_led": "RED"}
ACTION_SECURITY = {"fan": "OFF", "alarm": "ON", "servo": "CLOSED", "buzzer": "ON", "rgb_led": "RED"}

SCENARIOS = {
    0: "Routine",
    1: "Crowded",
    2: "Chemical",
    3: "Breach",
}


class AdaptiveGuardianFuzzy:
    """Fuzzy controller that converts intelligence signals into risk metadata."""

    def __init__(self):
        self.has_controller = fuzz is not None and ctrl is not None
        if self.has_controller:
            self._build_controller()

    def _build_controller(self):
        self.anomaly = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "art_anomaly")
        self.trend = ctrl.Antecedent(np.arange(-5, 5.1, 0.1), "rbf_trend")
        self.cluster = ctrl.Antecedent(np.arange(0, 3.1, 0.1), "som_cluster")
        self.spatial = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "gnn_spatial")

        self.scenario = ctrl.Consequent(np.arange(0, 4.1, 0.1), "scenario")
        self.risk = ctrl.Consequent(np.arange(0, 101, 1), "risk_level")

        self.anomaly["low"] = fuzz.gaussmf(self.anomaly.universe, 0.10, 0.12)
        self.anomaly["medium"] = fuzz.gaussmf(self.anomaly.universe, 0.45, 0.16)
        self.anomaly["high"] = fuzz.gaussmf(self.anomaly.universe, 0.85, 0.12)

        self.trend["negative"] = fuzz.gaussmf(self.trend.universe, -2.5, 1.2)
        self.trend["stable"] = fuzz.gaussmf(self.trend.universe, 0.0, 0.8)
        self.trend["positive"] = fuzz.gaussmf(self.trend.universe, 2.5, 1.2)

        self.cluster["normal"] = fuzz.gaussmf(self.cluster.universe, 0, 0.25)
        self.cluster["crowded"] = fuzz.gaussmf(self.cluster.universe, 1, 0.25)
        self.cluster["chemical"] = fuzz.gaussmf(self.cluster.universe, 2, 0.25)
        self.cluster["security"] = fuzz.gaussmf(self.cluster.universe, 3, 0.25)

        self.spatial["low"] = fuzz.gaussmf(self.spatial.universe, 0.10, 0.12)
        self.spatial["medium"] = fuzz.gaussmf(self.spatial.universe, 0.45, 0.18)
        self.spatial["high"] = fuzz.gaussmf(self.spatial.universe, 0.80, 0.14)

        self.scenario["routine"] = fuzz.trimf(self.scenario.universe, [0, 0, 1])
        self.scenario["crowded"] = fuzz.trimf(self.scenario.universe, [0.8, 1.4, 2.1])
        self.scenario["hazardous"] = fuzz.trimf(self.scenario.universe, [1.8, 2.5, 3.2])
        self.scenario["breach"] = fuzz.trimf(self.scenario.universe, [3, 4, 4])

        self.risk["safe"] = fuzz.trimf(self.risk.universe, [0, 0, 40])
        self.risk["warning"] = fuzz.trimf(self.risk.universe, [25, 50, 75])
        self.risk["critical"] = fuzz.trimf(self.risk.universe, [60, 100, 100])

        rules = []

        def add_rule(condition, scenario_term: str, risk_term: str):
            rules.append(ctrl.Rule(condition, self.scenario[scenario_term]))
            rules.append(ctrl.Rule(condition, self.risk[risk_term]))

        add_rule(self.cluster["normal"] & self.anomaly["low"] & self.trend["stable"], "routine", "safe")
        add_rule(self.cluster["normal"] & self.spatial["low"] & self.anomaly["low"], "routine", "safe")
        add_rule(self.cluster["crowded"] & self.anomaly["low"], "crowded", "warning")
        add_rule(self.cluster["crowded"] & self.spatial["medium"], "crowded", "warning")
        add_rule(self.cluster["chemical"] | self.anomaly["high"], "hazardous", "critical")
        add_rule(self.trend["positive"] & (self.anomaly["medium"] | self.spatial["medium"]), "hazardous", "critical")
        add_rule(self.cluster["security"], "breach", "critical")
        add_rule(self.cluster["security"] & (self.anomaly["medium"] | self.spatial["high"]), "breach", "critical")
        add_rule(self.anomaly["medium"] & self.trend["negative"], "crowded", "warning")
        add_rule(self.spatial["high"] & self.trend["positive"], "hazardous", "critical")

        self.control_system = ctrl.ControlSystem(rules)

    def predict(self, anomaly_score: float, trend_velocity: float, cluster_id: int, spatial_risk: float = 0.0):
        if not self.has_controller:
            return _fallback_predict(anomaly_score, trend_velocity, cluster_id, spatial_risk)

        try:
            sim = ctrl.ControlSystemSimulation(self.control_system)
            sim.input["art_anomaly"] = float(np.clip(anomaly_score, 0.0, 1.0))
            sim.input["rbf_trend"] = float(np.clip(trend_velocity, -5.0, 5.0))
            sim.input["som_cluster"] = float(np.clip(cluster_id, 0, 3))
            sim.input["gnn_spatial"] = float(np.clip(spatial_risk, 0.0, 1.0))
            sim.compute()

            scenario_value = float(sim.output.get("scenario", cluster_id))
            risk_value = float(sim.output.get("risk_level", 0.0))
            return _labels_from_scores(scenario_value, risk_value)
        except Exception:
            return _fallback_predict(anomaly_score, trend_velocity, cluster_id, spatial_risk)


def _labels_from_scores(scenario_value: float, risk_value: float) -> dict[str, Any]:
    if scenario_value < 1:
        scenario_label = "Routine"
    elif scenario_value < 2:
        scenario_label = "Crowded"
    elif scenario_value < 3:
        scenario_label = "Chemical"
    else:
        scenario_label = "Breach"

    if risk_value < 40:
        risk_label = "Safe"
    elif risk_value < 70:
        risk_label = "Warning"
    else:
        risk_label = "Critical"

    return {
        "scenario": scenario_label,
        "scenario_score": round(float(scenario_value), 2),
        "risk": risk_label,
        "risk_score": round(float(risk_value), 2),
    }


def _fallback_predict(anomaly_score: float, trend_velocity: float, cluster_id: int, spatial_risk: float = 0.0):
    cluster_id = int(np.clip(cluster_id, 0, 3))
    trend_pressure = max(float(trend_velocity), 0.0) * 8.0
    risk_value = (
        float(np.clip(anomaly_score, 0.0, 1.0)) * 35.0
        + float(np.clip(spatial_risk, 0.0, 1.0)) * 20.0
        + cluster_id * 16.0
        + trend_pressure
    )
    risk_value = float(np.clip(risk_value, 0.0, 100.0))
    return _labels_from_scores(float(cluster_id), risk_value)


def _action_from_decision(decision: dict[str, Any], motion: float, gas_level: float, temp_level: float) -> dict[str, str]:
    risk = decision["risk"]
    scenario = decision["scenario"]

    if risk == "Critical":
        if scenario in {"Breach", "Security"} or decision.get("scenario_id") == 3:
            return ACTION_SECURITY.copy()
        return ACTION_CRITICAL.copy()

    if risk == "Warning":
        return ACTION_WARNING.copy()

    return ACTION_SAFE.copy()


@lru_cache(maxsize=1)
def _engine() -> AdaptiveGuardianFuzzy:
    return AdaptiveGuardianFuzzy()


def decide(
    anomaly_score: float,
    trend: float,
    cluster_id: int,
    spatial_risk: float = 0.0,
    motion: float = 0.0,
    humidity_level: float = 0.0,
    gas_level: float = 0.0,
    light_level: float = 1.0,
    temp_level: float = 0.0,
    guard_risk_class: int = 0,
    guard_scenario_class: int = 0,
    guard_confidence: float = 0.0,
    guard_margin: float = 0.0,
    threshold_policy: dict[str, float] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Return the fuzzy baseline actuator command plus decision metadata."""
    decision = _engine().predict(anomaly_score, trend, cluster_id, spatial_risk)

    policy = threshold_policy or {}
    gas_warning = float(policy.get("gas_warning", 0.45))
    temp_warning = float(policy.get("temp_warning", 0.60))
    gas_danger = float(policy.get("gas_danger", 0.75))
    temp_danger = float(policy.get("temp_danger", 0.80))
    light_dark = float(policy.get("light_dark", 0.015))
    humidity_security = float(policy.get("humidity_security", 0.82))
    temp_security_max = float(policy.get("temp_security_max", 0.18))
    guard_warning_conf = float(policy.get("guard_warning_confidence", 0.46))
    guard_danger_conf = float(policy.get("guard_danger_confidence", 0.40))

    policy_danger = gas_level >= gas_danger or temp_level >= temp_danger
    policy_warning = gas_level >= gas_warning or temp_level >= temp_warning
    guard_chemical = guard_scenario_class == 2 and guard_confidence >= guard_danger_conf
    guard_security = guard_scenario_class == 3 and guard_confidence >= guard_danger_conf
    pir_security = motion > 0.5 and light_level <= light_dark and gas_level <= gas_warning
    security_signature = (
        pir_security
        or (
            light_level <= light_dark
            and humidity_level >= humidity_security
            and temp_level <= temp_security_max
            and gas_level <= gas_warning
            and guard_security
            and guard_confidence >= guard_danger_conf
        )
    )
    guard_danger = (guard_risk_class >= 2 and guard_confidence >= guard_danger_conf) or guard_chemical or guard_security
    guard_warning = guard_risk_class >= 1 and guard_confidence >= guard_warning_conf
    trend_spike = trend >= 3.2 and (gas_level >= gas_warning or temp_level >= temp_warning or spatial_risk >= 0.55)
    dark_trend_drop = (
        trend <= -3.2
        and light_level <= light_dark
        and humidity_level >= humidity_security
        and guard_security
    )
    strong_model_danger = cluster_id >= 2 and (anomaly_score >= 0.55 or spatial_risk >= 0.85)
    strong_model_warning = cluster_id >= 1 and (spatial_risk >= 0.55 or anomaly_score >= 0.40)

    if policy_danger or strong_model_danger or guard_danger or security_signature or trend_spike or dark_trend_drop:
        decision["risk"] = "Critical"
        decision["risk_score"] = max(float(decision.get("risk_score", 0.0)), 80.0)
        if guard_security or security_signature or dark_trend_drop:
            decision["scenario"] = "Breach"
            decision["scenario_id"] = 3
            decision["security_signature"] = True
        elif guard_chemical or policy_danger or trend_spike:
            decision["scenario"] = "Chemical"
            decision["scenario_id"] = 2
    elif policy_warning or strong_model_warning or guard_warning:
        decision["risk"] = "Warning" if decision.get("risk") == "Safe" else decision["risk"]
        decision["risk_score"] = max(float(decision.get("risk_score", 0.0)), 45.0)
        if guard_scenario_class == 1:
            decision["scenario"] = "Crowded"
            decision["scenario_id"] = 1
    elif cluster_id == 0:
        decision["risk"] = "Safe"
        decision["risk_score"] = min(float(decision.get("risk_score", 0.0)), 39.0)
        decision["scenario"] = "Routine"
        decision["scenario_id"] = 0

    decision["guard_risk_class"] = int(np.clip(guard_risk_class, 0, 2))
    decision["guard_scenario_class"] = int(np.clip(guard_scenario_class, 0, 3))
    decision["guard_confidence"] = round(float(np.clip(guard_confidence, 0.0, 1.0)), 4)
    decision["guard_margin"] = round(float(np.clip(guard_margin, 0.0, 1.0)), 4)
    decision["security_signature"] = bool(decision.get("security_signature", False))
    decision.setdefault("scenario_id", int(np.clip(cluster_id, 0, 3)))

    action = _action_from_decision(decision, motion, gas_level, temp_level)
    action["_decision"] = decision
    return action
