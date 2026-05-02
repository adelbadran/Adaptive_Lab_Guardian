# =============================================================================
#  Pipeline:
#    Sensor Data → GNN → PCA → RBF → SOM → ART2 → Fuzzy → RL → Action
# =============================================================================

import numpy as np
import torch
import os
import pickle

from ai.gnn import GATModel
from ai.rbf import step_rbf   

# =============================================================================
# COLORS
# =============================================================================
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# =============================================================================
# MODULE IMPORTS
# =============================================================================
def _try_import(name):
    import importlib
    try:
        mod = importlib.import_module(name)
        print(f"{GREEN}✔ Imported:{RESET} {name}")
        return mod
    except:
        print(f"{YELLOW}⚠ Missing:{RESET} {name}")
        return None


print(f"\n{BOLD}{CYAN}Loading Modules...{RESET}")

gnn_mod   = _try_import("ai.gnn")
pca_mod   = _try_import("ai.pca")
som_mod   = _try_import("ai.som")
art2_mod  = _try_import("ai.art2")
fuzzy_mod = _try_import("ai.fuzzy")
rl_mod    = _try_import("ai.rl")


# =============================================================================
# FEATURE ORDER
# =============================================================================
FEATURES = ["Temp_C", "Humidity_pct", "Gas_AQI", "Light_Lux", "Motion_Detected"]


# =============================================================================
# STEP FUNCTIONS
# =============================================================================

def step_pca(x):
    if pca_mod and hasattr(pca_mod, "transform"):
        return pca_mod.transform(x.reshape(1, -1)).flatten()
    return x


def step_som(x):
    gas = x[2]
    temp = x[0]

    if gas > 0.75 or temp > 0.8:
        return 2
    elif gas > 0.5:
        return 1
    return 0


def step_art2(x):
    return bool(np.any(x > 0.95))


def step_fuzzy(gas, state, motion):
    return {
        "fan": "ON" if state > 0 else "OFF",
        "alarm": "ON" if state == 2 else "OFF",
        "servo": "OPEN" if gas > 0.6 else "CLOSED",
        "buzzer": "ON" if state == 2 else "OFF",
        "rgb_led": "RED" if state == 2 else "GREEN"
    }


def step_rl(action):
    return action


# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline(sensor_data):

    print(f"\n{BOLD}{CYAN}--- NEW INPUT ---{RESET}")

    raw = np.array([sensor_data[c] for c in FEATURES])
    print("INPUT:", raw)

    # normalize
    x = np.clip(raw / np.array([50,100,1000,1000,1]), 0, 1)

    # ---------------- GNN ----------------
    x_gnn = x
    print("GNN:", x_gnn)

    # ---------------- PCA ----------------
    x_pca = step_pca(x_gnn)
    print("RBF INPUT (PCA):", x_pca)

    # ---------------- RBF ----------------
    rbf_out = step_rbf(x_pca)

    print(" RBF OUTPUT:", rbf_out)

    # ---------------- SOM ----------------
    state = step_som(x_pca)
    print("STATE:", state)

    # ---------------- ART2 ----------------
    anomaly = step_art2(x_pca)
    print("ANOMALY:", anomaly)

    # ---------------- FUZZY ----------------
    action = step_fuzzy(rbf_out["gas_trend"], state, x[4])
    print("ACTION:", action)

    # ---------------- RL ----------------
    action = step_rl(action)

    return action


