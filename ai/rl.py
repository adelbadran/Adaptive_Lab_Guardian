"""Reinforcement-learning refinement for the final decision stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


ACTION_TEMPLATES = [
    {"fan": "OFF", "alarm": "OFF", "servo": "CLOSED", "buzzer": "OFF", "rgb_led": "GREEN"},
    {"fan": "ON", "alarm": "OFF", "servo": "OPEN", "buzzer": "OFF", "rgb_led": "YELLOW"},
    {"fan": "ON", "alarm": "ON", "servo": "OPEN", "buzzer": "OFF", "rgb_led": "RED"},
    {"fan": "ON", "alarm": "ON", "servo": "OPEN", "buzzer": "ON", "rgb_led": "RED"},
]


def _action_level(action: dict[str, Any]) -> int:
    alarm = str(action.get("alarm", "OFF")).upper() == "ON"
    buzzer = str(action.get("buzzer", "OFF")).upper() == "ON"
    fan = str(action.get("fan", "OFF")).upper() == "ON"
    rgb = str(action.get("rgb_led", "GREEN")).upper()

    if alarm and buzzer:
        return 3
    if alarm or rgb == "RED":
        return 2
    if fan or rgb == "YELLOW":
        return 1
    return 0


def _state_index(cluster_id: int, context: dict[str, Any] | None = None) -> int:
    cluster = int(np.clip(cluster_id, 0, 3))
    if not context:
        return cluster

    risk_score = float(context.get("risk_score", 0.0))
    is_anomaly = bool(context.get("is_anomaly", False))
    if is_anomaly and risk_score >= 70:
        return 3
    return cluster


@dataclass
class QLearningDecisionAgent:
    """Small Q-table agent that learns whether to keep or intensify actions."""

    q_table: np.ndarray | None = None
    alpha: float = 0.25
    gamma: float = 0.90
    intensify_margin: float = 0.10

    def __post_init__(self):
        if self.q_table is None:
            self.q_table = np.zeros((4, len(ACTION_TEMPLATES)), dtype=float)
        else:
            self.q_table = np.asarray(self.q_table, dtype=float)
            if self.q_table.shape != (4, len(ACTION_TEMPLATES)):
                fixed = np.zeros((4, len(ACTION_TEMPLATES)), dtype=float)
                rows = min(fixed.shape[0], self.q_table.shape[0])
                cols = min(fixed.shape[1], self.q_table.shape[1])
                fixed[:rows, :cols] = self.q_table[:rows, :cols]
                self.q_table = fixed

    def update(self, cluster_id: int, baseline_action: dict[str, Any], reward: float, context: dict[str, Any] | None = None):
        state = _state_index(cluster_id, context)
        baseline_level = _action_level(baseline_action)

        old_value = self.q_table[state, baseline_level]
        target = float(reward) + self.gamma * float(np.max(self.q_table[state]))
        self.q_table[state, baseline_level] = old_value + self.alpha * (target - old_value)

        learned_level = int(np.argmax(self.q_table[state]))
        risk = str((context or {}).get("risk", "")).lower()
        if risk != "critical":
            learned_level = min(learned_level, baseline_level)

        should_intensify = learned_level > baseline_level and (
            self.q_table[state, learned_level] > self.q_table[state, baseline_level] + self.intensify_margin
        )

        if should_intensify:
            return ACTION_TEMPLATES[learned_level].copy()
        return {key: str(value).upper() for key, value in baseline_action.items()}


_AGENT = QLearningDecisionAgent()


def configure(q_table=None, alpha: float | None = None, gamma: float | None = None):
    """Configure the module-level agent from a saved Q-table."""
    global _AGENT
    _AGENT = QLearningDecisionAgent(
        q_table=q_table,
        alpha=_AGENT.alpha if alpha is None else alpha,
        gamma=_AGENT.gamma if gamma is None else gamma,
    )
    return _AGENT


def update(state: int, action_dict: dict[str, Any], reward: float = 0.0, context: dict[str, Any] | None = None):
    """Pipeline hook: update Q-values and return the final action."""
    return _AGENT.update(state, action_dict, reward, context=context)


def get_q_table() -> np.ndarray:
    return _AGENT.q_table.copy()
