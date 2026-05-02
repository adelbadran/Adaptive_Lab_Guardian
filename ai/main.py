# =============================================================================
#  Pipeline:
#    Sensor Data → GNN → PCA → RBF → SOM → ART2 → Fuzzy → RL → Action
# =============================================================================

import numpy as np
import pickle
import os
import torch
from ai.gnn import GATModel

GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

# =============================================================================
#  MODULE IMPORTS  (graceful fallback if teammate's file not ready yet)
# =============================================================================

def _try_import(module_name: str):
    """Try to import a module; return None with a warning if not found."""
    import importlib
    try:
        mod = importlib.import_module(module_name)
        print(f"  {GREEN}✔ Imported:{RESET} {module_name}.py")
        return mod
    except ImportError:
        print(f"  {YELLOW}⚠ Not ready yet (placeholder mode):{RESET} {module_name}.py")
        return None

print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
print(f"{BOLD}{CYAN}  Loading AI Modules...{RESET}")
print(f"{BOLD}{CYAN}{'='*60}{RESET}")

gnn_mod   = _try_import("ai.gnn")
pca_mod   = _try_import("ai.pca")
rbf_mod   = _try_import("rbf")
som_mod   = _try_import("som")
art2_mod  = _try_import("art2")
fuzzy_mod = _try_import("fuzzy")
rl_mod    = _try_import("rl")

# =============================================================================
#  MODEL LOADING  (load saved .pkl models from models/ folder)
# =============================================================================

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def _load_numpy(filename: str):
    """Load a NumPy .npy file; return None if not found yet."""
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        arr = np.load(path,allow_pickle=True)
        try:
            arr = arr.item()  # convert back to dict
        except:
            pass
        print(f"  {GREEN}✔ Loaded NumPy file:{RESET} {filename}")
        return arr
    else:
        print(f"  {YELLOW}⚠ NumPy file not found (placeholder):{RESET} {filename}")
        return None

# def _load_model(filename: str):
#     path = os.path.join(MODELS_DIR, filename)
#     if os.path.exists(path):
#         try:
#             import joblib
#             model = joblib.load(path)
#             print(f"  ✔ Loaded model (joblib): {filename}")
#             return model
#         except Exception:
#             try:
#                 with open(path, "rb") as f:
#                     model = pickle.load(f)
#                 print(f"  ✔ Loaded model (pickle): {filename}")
#                 return model
#             except Exception as e:
#                 print(f"  ⚠ Failed to load {filename}: {e}")
#                 return None
#     else:
#         print(f"  ⚠ Model not found (placeholder): {filename}")
#         return None

def _load_model(filename: str, model_class=None):
    path = os.path.join(MODELS_DIR, filename)

    if not os.path.exists(path):
        print(f"  ⚠ Model not found (placeholder): {filename}")
        return None

    try:
        # ── PyTorch (.pth) ─────────────────────────
        if filename.endswith(".pth"):
            state = torch.load(path, map_location="cpu")

            if isinstance(state, dict) and model_class is not None:
                model = model_class()
                model.load_state_dict(state)
                model.eval()
                print(f"  ✔ Loaded PyTorch model: {filename}")
                return model
            else:
                print(f"  ✔ Loaded full PyTorch model: {filename}")
                return state

        # ── sklearn (.pkl) ─────────────────────────
        elif filename.endswith(".pkl"):
            try:
                import joblib
                model = joblib.load(path)
                print(f"  ✔ Loaded model (joblib): {filename}")
                return model
            except Exception:
                with open(path, "rb") as f:
                    model = pickle.load(f)
                print(f"  ✔ Loaded model (pickle): {filename}")
                return model

        else:
            print(f"  ⚠ Unknown model type: {filename}")
            return None

    except Exception as e:
        print(f"  ⚠ Failed to load {filename}: {e}")
        return None
    
print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
print(f"{BOLD}{CYAN}  Loading Saved Models...{RESET}")
print(f"{BOLD}{CYAN}{'='*60}{RESET}")

gnn_model = _load_model("gnn.pth", model_class=GATModel)
pca_model  = _load_model("pca.pkl")
rbf_model  = _load_model("rbf.pkl")
som_model  = _load_model("som.pkl")
art2_model = _load_model("art2.pkl")
rl_table   = _load_numpy("rl_qtable.npy")   # RL Q-table (numpy array)
saved_attention = _load_numpy("gnn_attention.npy")

# Feature column order
FEATURE_COLS = ["Temp_C", "Humidity_pct", "Gas_AQI", "Light_Lux", "Motion_Detected"]

# State labels for SOM output
STATE_LABELS = {0: "Normal", 1: "Warning", 2: "Dangerous"}

# =============================================================================
#  STEP WRAPPERS
#  Each wrapper calls the real module function if available,
#  otherwise returns a safe dummy output so the pipeline never crashes.
# =============================================================================

# ── Step 1: GNN ───────────────────────────────────────────────────────────────

def create_edge_index(num_nodes=5):
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edges.append([i, j])
    return torch.tensor(edges, dtype=torch.long).t()

edge_index = create_edge_index()

def step_gnn(x: np.ndarray) -> np.ndarray:

    if gnn_model is None:
        # return saved attention (dataset) if no data yet become
            return x, saved_attention
    
    x_tensor = torch.tensor(x, dtype=torch.float).view(5, 1)
    batch = torch.zeros(5, dtype=torch.long)

    with torch.no_grad():
        emb, attn1, attn2 = gnn_model(x_tensor, edge_index, batch, return_attention=True)

    # Extract attention from first layer
    edge_idx, weights = attn1
    weights = weights.mean(dim=1)

    attention = {
        "edges": edge_idx.t().numpy().tolist(),
        "weights": weights.t().numpy().tolist()
    }

    return emb.numpy(), attention
# ── Step 2: PCA ──────────────────────────────────────────────────────────────
def step_pca(x: np.ndarray) -> np.ndarray:
    """
    Input  → high-dimensional vector (8 features) from GAT
    Output → lower-dimensional vector (3 features)
    """
    if pca_model and hasattr(pca_model, "transform"):
        return pca_model.transform(x.reshape(1, -1)).flatten()

    if pca_mod and hasattr(pca_mod, "transform"):
        return pca_mod.transform(x.reshape(1, -1)).flatten()

    # ── Placeholder: return input unchanged ──────────────────────────────────
    return x


# ── Step 3: RBF ──────────────────────────────────────────────────────────────
def step_rbf(x: np.ndarray) -> dict:
    """
    RBF Network: predict gas level and temperature trend.
    Person 4 implements: phi = compute_phi(x, centers, sigma); pred = phi.dot(W)
    Output: {"gas_pred": float, "temp_pred": float}
    """
    if rbf_model and hasattr(rbf_model, "predict"):
        pred = rbf_model.predict(x.reshape(1, -1)).flatten()
        return {"gas_pred": float(pred[0]), "temp_pred": float(pred[1])}

    if rbf_mod and hasattr(rbf_mod, "predict"):
        return rbf_mod.predict(x)

    # ── Placeholder: pass-through raw sensor values ──────────────────────────
    # x[2] = Gas_AQI position in GNN output (approx), x[0] = Temp
    return {"gas_pred": float(x[2]), "temp_pred": float(x[0])}


# ── Step 4: SOM ──────────────────────────────────────────────────────────────
def step_som(x: np.ndarray) -> int:
    """
    SOM: classify system state → 0=Normal, 1=Warning, 2=Dangerous.
    Person 5 implements: state = som.predict(x_pca)
    Output: int (0 / 1 / 2)
    """
    if som_model and hasattr(som_model, "predict"):
        return int(som_model.predict(x.reshape(1, -1)))

    if som_mod and hasattr(som_mod, "predict"):
        return int(som_mod.predict(x))

    # ── Placeholder: rule-based state from gas + temp values (index 2 & 0) ──
    gas  = x[2]
    temp = x[0]
    if gas > 0.75 or temp > 0.80:
        return 2   # Dangerous
    elif gas > 0.45 or temp > 0.55:
        return 1   # Warning
    return 0       # Normal


# ── Step 5: ART2 ─────────────────────────────────────────────────────────────
def step_art2(x: np.ndarray) -> bool:
    """
    ART2: detect new/unseen patterns (anomaly detection).
    Person 5 implements: is_new = distance(x, known_patterns) > threshold
    Output: True = new unseen pattern, False = known pattern
    """
    if art2_model and hasattr(art2_model, "is_new_pattern"):
        return bool(art2_model.is_new_pattern(x))

    if art2_mod and hasattr(art2_mod, "is_new_pattern"):
        return bool(art2_mod.is_new_pattern(x))

    # ── Placeholder: flag if any feature is in extreme range ─────────────────
    return bool(np.any(x > 0.95) or np.any(x < 0.02))


# ── Step 6: Fuzzy Logic ──────────────────────────────────────────────────────
def step_fuzzy(gas_pred: float, state: int, motion: float) -> dict:
    """
    Fuzzy Logic: convert predictions → human-like action decisions.
    Person 6 implements fuzzy rules.
    Output: {"fan": str, "alarm": str, "servo": str, "buzzer": str, "rgb_led": str}
    """
    if fuzzy_mod and hasattr(fuzzy_mod, "decide"):
        return fuzzy_mod.decide(gas_pred, state, motion)

    # ── Placeholder: rule-based fuzzy decisions ───────────────────────────────
    fan    = "OFF"
    alarm  = "OFF"
    servo  = "CLOSED"
    buzzer = "OFF"
    rgb    = "GREEN"

    if state == 2:                          # Dangerous
        alarm  = "ON"
        buzzer = "ON"
        fan    = "ON"
        servo  = "OPEN"
        rgb    = "RED"
    elif state == 1:                        # Warning
        fan    = "ON"
        rgb    = "YELLOW"
        if gas_pred > 0.6:
            alarm = "ON"
            servo = "OPEN"
    else:                                   # Normal
        rgb = "GREEN"
        if motion > 0.5:
            servo = "OPEN"

    return {
        "fan":     fan,
        "alarm":   alarm,
        "servo":   servo,
        "buzzer":  buzzer,
        "rgb_led": rgb,
    }


# ── Step 7: Reinforcement Learning ───────────────────────────────────────────
def step_rl(state: int, action_dict: dict, reward: float = 0.0) -> dict:
    """
    RL (Q-learning): improve decisions over time.
    Person 6 implements: Q[state, action] += reward
    Output: (possibly refined) action dict
    """
    if rl_mod and hasattr(rl_mod, "update"):
        return rl_mod.update(state, action_dict, reward)

    # ── Placeholder: return action unchanged (RL update is a side-effect) ────
    # In real implementation, Q-table lookup might override action
    return action_dict


# =============================================================================
#  REWARD HELPER  (used internally by RL step)
# =============================================================================

def _compute_reward(state: int, action_dict: dict, is_anomaly: bool) -> float:
    """
    Simple reward signal for RL:
    +1  if correct action taken for the state
    -1  if alarm not raised during dangerous state
     0  otherwise
    """
    if state == 2 and action_dict["alarm"] == "ON":
        return 1.0
    if state == 2 and action_dict["alarm"] == "OFF":
        return -1.0
    if state == 0 and action_dict["alarm"] == "OFF":
        return 0.5
    if is_anomaly:
        return -0.5   # penalise unknown situations slightly
    return 0.0


# =============================================================================
#  MAIN PIPELINE FUNCTION
# =============================================================================

def run_pipeline(sensor_data: dict, scaler=None, verbose: bool = True) -> dict:
    """
    Run the full AI pipeline on one real-time sensor reading.

    Parameters
    ----------
    sensor_data : dict  — e.g. {"Temp_C": 33.5, "Humidity_pct": 62, ...}
    scaler      : fitted MinMaxScaler from preprocessing.py (optional)
    verbose     : bool  — print step-by-step logs

    Returns
    -------
    result : dict — final action decisions + pipeline metadata
    """

    def log(step: str, msg: str):
        if verbose:
            print(f"  {BOLD}{BLUE}[{step}]{RESET} {msg}")

    if verbose:
        print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
        print(f"{BOLD}{CYAN}  🚀 AI Pipeline — New Sensor Reading{RESET}")
        print(f"{BOLD}{CYAN}{'='*60}{RESET}")

    # ── 0. Validate & convert input ──────────────────────────────────────────
    missing = [c for c in FEATURE_COLS if c not in sensor_data]
    if missing:
        raise ValueError(f"Missing sensor keys: {missing}")

    raw = np.array([sensor_data[c] for c in FEATURE_COLS], dtype=float)
    log("INPUT", f"Raw sensor → {dict(zip(FEATURE_COLS, raw.round(3)))}")

    # ── Scale if scaler is provided (from preprocessing.py) ──────────────────
    if scaler is not None:
        x = scaler.transform(raw.reshape(1, -1)).flatten()
        log("SCALE", f"MinMaxScaler applied → {x.round(4)}")
    else:
        # Soft normalise with estimated ranges if no scaler available
        ranges = np.array([50.0, 100.0, 1000.0, 1000.0, 1.0])   # approx max per col
        x = np.clip(raw / ranges, 0.0, 1.0)
        log("SCALE", f"Fallback normalise → {x.round(4)}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — GNN
    # ─────────────────────────────────────────────────────────────────────────
    x_gnn, attention = step_gnn(x)
    # Re-normalise after GNN dot product to keep values in [0, 1]
    x_gnn = np.clip(x_gnn / (x_gnn.max() + 1e-9), 0.0, 1.0)
    log("GNN ", f"Sensor relationships applied → {x_gnn.round(4)}")

    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — PCA
    # ─────────────────────────────────────────────────────────────────────────
    x_pca = step_pca(x_gnn)
    log("PCA ", f"Dimensionality reduced → shape {x_pca.shape}, values {x_pca.round(4)}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — RBF
    # ─────────────────────────────────────────────────────────────────────────
    rbf_out   = step_rbf(x_pca)
    gas_pred  = rbf_out["gas_pred"]
    temp_pred = rbf_out["temp_pred"]
    log("RBF ", f"gas_pred={gas_pred:.4f}  temp_pred={temp_pred:.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4 — SOM
    # ─────────────────────────────────────────────────────────────────────────
    state       = step_som(x_pca)
    state_label = STATE_LABELS.get(state, "Unknown")
    log("SOM ", f"System state → {state} ({state_label})")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5 — ART2
    # ─────────────────────────────────────────────────────────────────────────
    is_anomaly = step_art2(x_pca)
    anomaly_str = f"{RED}⚠ NEW UNSEEN PATTERN{RESET}" if is_anomaly else f"{GREEN}Known pattern{RESET}"
    log("ART2", f"Anomaly detection → {anomaly_str}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6 — Fuzzy Logic
    # ─────────────────────────────────────────────────────────────────────────
    motion     = x[4]   # Motion_Detected (scaled)
    action     = step_fuzzy(gas_pred, state, motion)
    log("FUZZ", f"Fuzzy decision → {action}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7 — RL update
    # ─────────────────────────────────────────────────────────────────────────
    reward = _compute_reward(state, action, is_anomaly)
    action = step_rl(state, action, reward)
    log("RL  ", f"Reward={reward:+.1f}  →  Final action → {action}")

    # ─────────────────────────────────────────────────────────────────────────
    # FINAL OUTPUT
    # ─────────────────────────────────────────────────────────────────────────
    result = {
        # ── actuator commands (sent to ESP32 via mqtt_client.py) ──
        **action,
        # ── metadata (used by dashboard/app.py) ──────────────────
        "_meta": {
            "state":       state,
            "state_label": state_label,
            "gas_pred":    round(gas_pred,  4),
            "temp_pred":   round(temp_pred, 4),
            "is_anomaly":  is_anomaly,
            "reward":      reward,
            "attention":   attention 
        }
    }

    if verbose:
        print(f"\n{BOLD}{'─'*60}{RESET}")
        print(f"{BOLD}  ✅ FINAL ACTION:{RESET}")
        for k, v in action.items():
            colour = RED if v in ("ON", "OPEN") else GREEN
            print(f"     {k:<10}: {colour}{BOLD}{v}{RESET}")
        print(f"\n{BOLD}  📊 PIPELINE METADATA:{RESET}")
        meta = result["_meta"]
        print(f"     State       : {meta['state']} ({meta['state_label']})")
        print(f"     Gas pred    : {meta['gas_pred']}")
        print(f"     Temp pred   : {meta['temp_pred']}")
        print(f"     Anomaly     : {meta['is_anomaly']}")
        print(f"     RL reward   : {meta['reward']:+.1f}")
        print(f"{BOLD}{CYAN}{'='*60}{RESET}\n")

    return result


# =============================================================================
#  BATCH PIPELINE  (for testing on historical data)
# =============================================================================

def run_batch(df, scaler=None, verbose: bool = False) -> list:
    """
    Run the pipeline on an entire DataFrame (e.g. from preprocessing.py).
    Used for offline evaluation and debugging.

    Parameters
    ----------
    df      : pd.DataFrame with FEATURE_COLS columns
    scaler  : fitted MinMaxScaler (optional)
    verbose : bool — print per-row logs

    Returns
    -------
    list of result dicts
    """
    results = []
    for i, row in df.iterrows():
        sensor_data = {col: row[col] for col in FEATURE_COLS if col in row}
        result = run_pipeline(sensor_data, scaler=scaler, verbose=verbose)
        results.append(result)
    return results


# =============================================================================
#  STANDALONE TEST  (run:  python main.py)
# =============================================================================

if __name__ == "__main__":

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  STANDALONE TEST — 3 sensor scenarios{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    test_cases = [
        {
            "label": "🟢 Normal lab conditions",
            "data": {
                "Temp_C": 22.0,
                "Humidity_pct": 45.0,
                "Gas_AQI": 120.0,
                "Light_Lux": 400.0,
                "Motion_Detected": 0,
            },
        },
        {
            "label": "🟡 Warning — elevated gas + temp",
            "data": {
                "Temp_C": 33.5,
                "Humidity_pct": 62.0,
                "Gas_AQI": 520.0,
                "Light_Lux": 240.0,
                "Motion_Detected": 1,
            },
        },
        {
            "label": "🔴 Dangerous — high gas + dark + motion",
            "data": {
                "Temp_C": 42.0,
                "Humidity_pct": 80.0,
                "Gas_AQI": 850.0,
                "Light_Lux": 10.0,
                "Motion_Detected": 1,
            },
        },
    ]

    for tc in test_cases:
        print(f"\n{BOLD}{YELLOW}── {tc['label']} ──{RESET}")
        result = run_pipeline(tc["data"], scaler=None, verbose=True)
