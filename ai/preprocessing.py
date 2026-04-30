import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils import resample

GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def preprocess_data(csv_path: str, oversample_train: bool = False):
    """
    Full preprocessing pipeline for the Adaptive Lab Guardian dataset.

    Parameters
    ----------
    csv_path       : str   — path to Adaptive_Lab_Guardian.csv
    oversample_train: bool — if True, apply random oversampling on training set
                             to balance minority classes (default = False)

    Returns
    -------
    X_train        : np.ndarray — scaled training features
    X_test         : np.ndarray — scaled test features
    y_train        : np.ndarray — encoded training labels
    y_test         : np.ndarray — encoded test labels
    scaler         : fitted MinMaxScaler
    le             : fitted LabelEncoder
    feature_cols   : list[str]  — feature column names
    """

    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  Smart Adaptive Environment — Preprocessing Pipeline{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — Load CSV
    # ─────────────────────────────────────────────────────────────────────────
    print(f"{BOLD}[1] Loading dataset...{RESET}")
    df = pd.read_csv(csv_path)
    print(f"    {GREEN}✔ Loaded:{RESET} {df.shape[0]} rows × {df.shape[1]} columns")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — Validate required columns
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}[2] Validating columns...{RESET}")
    required_cols = [
        "Timestamp", "Temp_C", "Humidity_pct",
        "Gas_AQI", "Light_Lux", "Motion_Detected", "True_Scenario"
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{RED}✘ Missing columns: {missing_cols}{RESET}")
    print(f"    {GREEN}✔ All required columns present{RESET}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — Convert Timestamp to datetime
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}[3] Parsing Timestamp...{RESET}")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    n_bad_ts = df["Timestamp"].isna().sum()
    if n_bad_ts > 0:
        print(f"    {YELLOW}⚠ Dropping {n_bad_ts} rows with unparseable Timestamp{RESET}")
        df = df.dropna(subset=["Timestamp"])
    # Keep the data in chronological order (time-series — DO NOT shuffle)
    df = df.sort_values("Timestamp").reset_index(drop=True)
    print(f"    {GREEN}✔ Timestamp range:{RESET} {df['Timestamp'].iloc[0]}  →  {df['Timestamp'].iloc[-1]}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4 — Select feature columns (NO True_Scenario as input)
    # ─────────────────────────────────────────────────────────────────────────
    feature_cols = ["Temp_C", "Humidity_pct", "Gas_AQI", "Light_Lux", "Motion_Detected"]
    print(f"\n{BOLD}[4] Feature columns:{RESET} {feature_cols}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5 — Check and handle missing values
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}[5] Checking missing values...{RESET}")
    missing = df[feature_cols + ["True_Scenario"]].isnull().sum()
    total_missing = missing.sum()
    if total_missing == 0:
        print(f"    {GREEN}✔ No missing values found{RESET}")
    else:
        print(f"    {YELLOW}⚠ Missing values detected:{RESET}")
        print(missing[missing > 0].to_string())
        # Strategy: forward-fill then back-fill (safe for time-series)
        df[feature_cols]      = df[feature_cols].ffill().bfill()
        df["True_Scenario"]   = df["True_Scenario"].ffill().bfill()
        print(f"    {GREEN}✔ Missing values handled with forward/back-fill{RESET}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6 — Encode True_Scenario → numeric labels
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}[6] Encoding target labels...{RESET}")
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["True_Scenario"])
    print(f"    {GREEN}✔ Classes:{RESET} {list(le.classes_)}  →  mapped to  {list(range(len(le.classes_)))}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7 — Class distribution (before split)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}[7] Class distribution (full dataset):{RESET}")
    counts = df["True_Scenario"].value_counts().sort_index()
    for cls, cnt in counts.items():
        pct = 100 * cnt / len(df)
        bar = "█" * int(pct / 2)
        print(f"    Scenario {cls}: {cnt:5d} samples  ({pct:5.1f}%)  {CYAN}{bar}{RESET}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 8 — Temporal train/test split  (80 / 20 — NO shuffle)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}[8] Temporal train/test split (80/20 — no shuffle)...{RESET}")
    split_idx = int(len(df) * 0.80)

    X = df[feature_cols].values
    y = df["label"].values

    X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
    y_train,     y_test      = y[:split_idx], y[split_idx:]

    print(f"    {GREEN}✔ Train:{RESET} {len(X_train_raw)} samples  |  "
          f"{GREEN}Test:{RESET} {len(X_test_raw)} samples")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 9 — (Optional) Oversample minority classes in TRAIN ONLY
    # ─────────────────────────────────────────────────────────────────────────
    if oversample_train:
        print(f"\n{BOLD}[9] Applying oversampling on training set...{RESET}")
        print(f"    {YELLOW}⚠ Note: oversampling applied to training only.{RESET}")
        print(f"    {YELLOW}  Dataset is intentionally imbalanced — test set is NOT touched.{RESET}")
        train_df = pd.DataFrame(X_train_raw, columns=feature_cols)
        train_df["label"] = y_train

        max_count = train_df["label"].value_counts().max()
        balanced_parts = []
        for cls_id in train_df["label"].unique():
            cls_df = train_df[train_df["label"] == cls_id]
            if len(cls_df) < max_count:
                cls_df = resample(cls_df, replace=True,
                                  n_samples=max_count, random_state=42)
            balanced_parts.append(cls_df)

        train_balanced = pd.concat(balanced_parts).sample(
            frac=1, random_state=42).reset_index(drop=True)
        X_train_raw = train_balanced[feature_cols].values
        y_train     = train_balanced["label"].values
        print(f"    {GREEN}✔ Balanced train size:{RESET} {len(X_train_raw)} samples")
    else:
        print(f"\n{BOLD}[9] Oversampling:{RESET} {YELLOW}skipped (keeping original imbalance){RESET}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 10 — Scale features with MinMaxScaler (fit on TRAIN only)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}[10] Scaling features with MinMaxScaler...{RESET}")
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)   # fit on train ONLY
    X_test  = scaler.transform(X_test_raw)         # transform test with same scaler
    print(f"    {GREEN}✔ Scaler fitted on training set and applied to both splits{RESET}")
    print(f"    Feature ranges after scaling (train):")
    for i, col in enumerate(feature_cols):
        print(f"      {col}: [{X_train[:, i].min():.3f}, {X_train[:, i].max():.3f}]")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 11 — Final summary
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}  ✅ Preprocessing complete — Summary{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"  Dataset shape       : {df.shape}")
    print(f"  Features used       : {feature_cols}")
    print(f"  X_train shape       : {X_train.shape}")
    print(f"  X_test  shape       : {X_test.shape}")
    print(f"  y_train distribution: { {str(le.classes_[k]): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))} }")
    print(f"  y_test  distribution: { {str(le.classes_[k]): int(v) for k, v in zip(*np.unique(y_test,  return_counts=True))} }")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # RETURNS (everything the AI pipeline needs)
    # ─────────────────────────────────────────────────────────────────────────
    return X_train, X_test, y_train, y_test, scaler, le, feature_cols


# =============================================================================
#  STANDALONE TEST  (run:  python preprocessing.py)
# =============================================================================
if __name__ == "__main__":
    CSV_PATH = r"C:\Adaptive Lab Guardian\data\Adaptive_Lab_Guardian.csv"  

    X_train, X_test, y_train, y_test, scaler, le, feature_cols = preprocess_data(
        csv_path=CSV_PATH,
        oversample_train=False
    )

    print("X_train dtype :", X_train.dtype)
    print("y_train dtype :", y_train.dtype)
    print("Scaler type   :", type(scaler).__name__)
    print("LabelEncoder  :", le.classes_)
    print("Feature cols  :", feature_cols)
