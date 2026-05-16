"""Genetic optimisation stage for Adaptive Lab Guardian.

The GA belongs to the evolution block in the project diagram. It tunes simple
decision thresholds from labelled historical data or reward logs, then those
thresholds can be used to update fuzzy/RL configuration offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from ai.preprocessing import preprocess_data
except Exception:  # pragma: no cover - optional training dependency
    preprocess_data = None


@dataclass
class ThresholdPolicy:
    gas_warning: float
    temp_warning: float
    gas_danger: float
    temp_danger: float

    def as_dict(self) -> dict[str, float]:
        return {
            "gas_warning": round(float(self.gas_warning), 4),
            "temp_warning": round(float(self.temp_warning), 4),
            "gas_danger": round(float(self.gas_danger), 4),
            "temp_danger": round(float(self.temp_danger), 4),
        }


class GA:
    """Tune warning/danger thresholds for the fuzzy baseline."""

    def __init__(
        self,
        pop_size: int = 18,
        generations: int = 20,
        mutation_rate: float = 0.25,
        false_negative_cost: float = 0.65,
        false_positive_cost: float = 0.55,
        over_alert_cost: float = 0.30,
        seed: int = 42,
    ):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.false_negative_cost = false_negative_cost
        self.false_positive_cost = false_positive_cost
        self.over_alert_cost = over_alert_cost
        self.rng = np.random.default_rng(seed)

    def init_population(self) -> list[ThresholdPolicy]:
        population = []
        for _ in range(self.pop_size):
            gas_warning = self.rng.uniform(0.25, 0.62)
            temp_warning = self.rng.uniform(0.45, 0.72)
            gas_danger = self.rng.uniform(max(gas_warning + 0.14, 0.62), 0.96)
            temp_danger = self.rng.uniform(max(temp_warning + 0.14, 0.72), 0.99)
            population.append(ThresholdPolicy(gas_warning, temp_warning, gas_danger, temp_danger))
        return population

    @staticmethod
    def _normalise_labels(y: Iterable[int]) -> np.ndarray:
        y = np.asarray(y)
        numeric = y.astype(float)
        if np.all(np.isin(numeric, [1, 2, 3, 4])):
            raw = numeric.astype(int)
            return np.where(raw >= 3, 2, np.where(raw == 2, 1, 0)).astype(int)

        unique = np.unique(y)
        if len(unique) <= 3:
            mapping = {label: idx for idx, label in enumerate(sorted(unique))}
            return np.array([mapping[value] for value in y], dtype=int)

        low = np.percentile(unique, 34)
        high = np.percentile(unique, 67)
        return np.where(y >= high, 2, np.where(y >= low, 1, 0)).astype(int)

    @staticmethod
    def predict_classes(X: np.ndarray, policy: ThresholdPolicy) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        temp = X[:, 0]
        gas = X[:, 2]

        danger = (gas >= policy.gas_danger) | (temp >= policy.temp_danger)
        warning = (gas >= policy.gas_warning) | (temp >= policy.temp_warning)
        return np.where(danger, 2, np.where(warning, 1, 0)).astype(int)

    def fitness(self, policy: ThresholdPolicy, X: np.ndarray, y: np.ndarray) -> float:
        target = self._normalise_labels(y)
        pred = self.predict_classes(X, policy)

        recalls = []
        precisions = []
        for cls in (0, 1, 2):
            tp = np.sum((target == cls) & (pred == cls))
            fn = np.sum((target == cls) & (pred != cls))
            fp = np.sum((target != cls) & (pred == cls))
            recalls.append(tp / max(tp + fn, 1))
            precisions.append(tp / max(tp + fp, 1))

        balanced_recall = float(np.mean(recalls))
        macro_precision = float(np.mean(precisions))
        false_negative = float(np.mean((target == 2) & (pred < 2)))
        false_positive = float(np.mean((target < 2) & (pred == 2)))
        over_alert = float(np.mean((target == 0) & (pred > 0)))
        warning_miss = float(np.mean((target == 1) & (pred == 0)))

        return (
            0.55 * balanced_recall
            + 0.45 * macro_precision
            - self.false_negative_cost * false_negative
            - self.false_positive_cost * false_positive
            - self.over_alert_cost * over_alert
            - 0.20 * warning_miss
        )

    def selection(self, population: list[ThresholdPolicy], scores: list[float]) -> list[ThresholdPolicy]:
        keep = max(2, self.pop_size // 3)
        order = np.argsort(scores)[::-1]
        return [population[i] for i in order[:keep]]

    def crossover(self, p1: ThresholdPolicy, p2: ThresholdPolicy) -> ThresholdPolicy:
        values = []
        for a, b in zip(p1.as_dict().values(), p2.as_dict().values()):
            values.append(a if self.rng.random() < 0.5 else b)
        return ThresholdPolicy(*values)

    def mutate(self, policy: ThresholdPolicy) -> ThresholdPolicy:
        values = np.array(list(policy.as_dict().values()), dtype=float)
        for i in range(len(values)):
            if self.rng.random() < self.mutation_rate:
                values[i] += self.rng.normal(0.0, 0.05)

        values = np.clip(values, 0.05, 0.98)
        values[2] = max(values[2], values[0] + 0.08)
        values[3] = max(values[3], values[1] + 0.08)
        values = np.clip(values, 0.05, 0.98)
        return ThresholdPolicy(*values)

    def run(self, X_train, y_train, X_test=None, y_test=None, verbose: bool = True):
        X_eval = np.asarray(X_train if X_test is None else X_test, dtype=float)
        y_eval = np.asarray(y_train if y_test is None else y_test)

        population = self.init_population()
        best_policy = population[0]
        best_score = float("-inf")

        for generation in range(self.generations):
            scores = [self.fitness(policy, X_eval, y_eval) for policy in population]
            gen_best_idx = int(np.argmax(scores))
            if scores[gen_best_idx] > best_score:
                best_score = float(scores[gen_best_idx])
                best_policy = population[gen_best_idx]

            if verbose:
                print(f"Generation {generation:02d}: best_score={best_score:.4f} policy={best_policy.as_dict()}")

            parents = self.selection(population, scores)
            next_population = parents.copy()
            while len(next_population) < self.pop_size:
                p1 = parents[int(self.rng.integers(0, len(parents)))]
                p2 = parents[int(self.rng.integers(0, len(parents)))]
                next_population.append(self.mutate(self.crossover(p1, p2)))
            population = next_population

        return best_policy.as_dict(), best_score


if __name__ == "__main__":
    if preprocess_data is None:
        raise SystemExit("Install requirements.txt before running GA on the dataset.")
    csv_path = Path(__file__).resolve().parents[1] / "data" / "Adaptive_Lab_Guardian.csv"
    X_train, X_test, y_train, y_test, *_ = preprocess_data(str(csv_path))
    best, score = GA(pop_size=18, generations=20).run(X_train, y_train, X_test, y_test)
    print("Best policy:", best)
    print("Best score:", round(score, 4))
