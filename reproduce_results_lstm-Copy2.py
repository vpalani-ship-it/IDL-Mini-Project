"""
reproduce_results_lstm.py
=========================
Reproduces all key figures and metrics from:
  "RUL Prediction using LSTM with Piecewise Linear Degradation Model
   on NASA CMAPSS FD001"

Usage:
    python reproduce_results_lstm.py

Requirements:
    pip install tensorflow numpy pandas scikit-learn matplotlib gdown

What this script does (NO retraining):
    1. Downloads the trained .h5 checkpoint from Google Drive (if not present)
    2. Loads and preprocesses CMAPSS FD001 data identically to the notebook
    3. Loads the saved Keras model
    4. Runs inference and prints all reported metrics (RMSE, S-score)
    5. Saves all key figures to ./figures/
"""

# ─────────────────────────────────────────────────────────────
# 0. CONFIGURATION  ← paste your Google Drive file ID here
# ─────────────────────────────────────────────────────────────
GDRIVE_FILE_ID  = "1dfOLxevVohwTdwG4159xS2Gt1PHcMfRE"
CHECKPOINT_PATH = "FD001_LSTM_piecewise.h5"     # local filename to save / load
DATA_DIR        = "CMAPSSData"                  # folder with train/test/RUL txts
# ─────────────────────────────────────────────────────────────

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

np.random.seed(34)   # same seed as notebook

os.makedirs("figures", exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 1. DOWNLOAD CHECKPOINT FROM GOOGLE DRIVE
# ─────────────────────────────────────────────────────────────
def download_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint already exists at '{CHECKPOINT_PATH}'. Skipping download.")
        return
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required for auto-download.\n"
            "Install it with:  pip install gdown\n"
            "Or manually place the .h5 file in this directory."
        )
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    print(f"Downloading checkpoint from Google Drive -> '{CHECKPOINT_PATH}' ...")
    gdown.download(url, CHECKPOINT_PATH, quiet=False)
    print("Download complete.")

download_checkpoint()


# ─────────────────────────────────────────────────────────────
# 2. DATA PREPROCESSING  (identical to notebook)
# ─────────────────────────────────────────────────────────────

# ── Hyperparameters (must match training) ────────────────────
WINDOW_LENGTH    = 20
SHIFT            = 1
EARLY_RUL        = 100
NUM_TEST_WINDOWS = 5          # windows averaged per engine
COLUMNS_TO_DROP  = [0,1,2,3,4,5,9,10,14,20,22,23]

def process_targets(data_length, early_rul=None):
    if early_rul is None:
        return np.arange(data_length - 1, -1, -1)
    early_rul_duration = data_length - early_rul
    if early_rul_duration <= 0:
        return np.arange(data_length - 1, -1, -1)
    return np.append(
        early_rul * np.ones(shape=(early_rul_duration,)),
        np.arange(early_rul - 1, -1, -1)
    )

def process_input_data_with_targets(input_data, target_data=None,
                                    window_length=1, shift=1):
    num_batches  = int(np.floor((len(input_data) - window_length) / shift)) + 1
    num_features = input_data.shape[1]
    output_data  = np.full((num_batches, window_length, num_features), np.nan)
    if target_data is None:
        for b in range(num_batches):
            output_data[b] = input_data[b*shift : b*shift + window_length]
        return output_data
    output_targets = np.full(num_batches, np.nan)
    for b in range(num_batches):
        output_data[b]    = input_data[b*shift : b*shift + window_length]
        output_targets[b] = target_data[b*shift + (window_length - 1)]
    return output_data, output_targets

def process_test_data(test_data_for_engine, window_length, shift, num_test_windows=1):
    max_batches = int(np.floor((len(test_data_for_engine) - window_length) / shift)) + 1
    if max_batches < num_test_windows:
        required_len = (max_batches - 1) * shift + window_length
        batched = process_input_data_with_targets(
            test_data_for_engine[-required_len:], None, window_length, shift)
        return batched, max_batches
    required_len = (num_test_windows - 1) * shift + window_length
    batched = process_input_data_with_targets(
        test_data_for_engine[-required_len:], None, window_length, shift)
    return batched, num_test_windows


print("\n── Loading & preprocessing CMAPSS FD001 data ──")

train_data = pd.read_csv(os.path.join(DATA_DIR, "train_FD001.txt"),
                         sep=r"\s+", header=None)
test_data  = pd.read_csv(os.path.join(DATA_DIR, "test_FD001.txt"),
                         sep=r"\s+", header=None)
true_rul   = pd.read_csv(os.path.join(DATA_DIR, "RUL_FD001.txt"),
                         sep=r"\s+", header=None)

# Keep engine-ID columns before scaling
train_engine_col = train_data[0].copy()
test_engine_col  = test_data[0].copy()

# StandardScaler fit on train, applied to both
scaler     = StandardScaler()
train_scaled = scaler.fit_transform(train_data.drop(columns=COLUMNS_TO_DROP))
test_scaled  = scaler.transform(test_data.drop(columns=COLUMNS_TO_DROP))

train_data = pd.DataFrame(np.c_[train_engine_col, train_scaled])
test_data  = pd.DataFrame(np.c_[test_engine_col,  test_scaled])

num_train_engines = len(train_data[0].unique())
num_test_engines  = len(test_data[0].unique())

# ── Process test data ─────────────────────────────────────────
processed_test_data      = []
num_test_windows_list    = []

for i in np.arange(1, num_test_engines + 1):
    temp = test_data[test_data[0] == i].drop(columns=[0]).values
    if len(temp) < WINDOW_LENGTH:
        raise AssertionError(f"Test engine {i} has fewer rows than window_length={WINDOW_LENGTH}")
    batched, n_windows = process_test_data(temp, WINDOW_LENGTH, SHIFT, NUM_TEST_WINDOWS)
    processed_test_data.append(batched)
    num_test_windows_list.append(n_windows)

processed_test_data = np.concatenate(processed_test_data)
true_rul = true_rul[0].values

print(f"Processed test data shape : {processed_test_data.shape}")
print(f"True RUL shape            : {true_rul.shape}")


# ─────────────────────────────────────────────────────────────
# 3. LOAD KERAS MODEL
# ─────────────────────────────────────────────────────────────
import tensorflow as tf

print(f"\n── Loading checkpoint: {CHECKPOINT_PATH} ──")
model = tf.keras.models.load_model(CHECKPOINT_PATH,compile=False)
model.summary()
print("Model loaded successfully.")


# ─────────────────────────────────────────────────────────────
# 4. INFERENCE
# ─────────────────────────────────────────────────────────────
print("\n── Running inference ──")
rul_pred = model.predict(processed_test_data, batch_size=128).reshape(-1)

# Split predictions back to per-engine groups
preds_per_engine = np.split(rul_pred, np.cumsum(num_test_windows_list)[:-1])

# Averaged prediction (primary metric, matches notebook)
mean_pred = np.array([
    np.average(preds, weights=np.repeat(1/n, n))
    for preds, n in zip(preds_per_engine, num_test_windows_list)
])

# Last-window-only prediction (secondary metric)
indices_last = np.cumsum(num_test_windows_list) - 1
last_pred    = np.concatenate(preds_per_engine)[indices_last]


# ─────────────────────────────────────────────────────────────
# 5. METRICS
# ─────────────────────────────────────────────────────────────
def compute_s_score(rul_true, rul_pred):
    diff = np.asarray(rul_pred) - np.asarray(rul_true)
    return float(np.sum(np.where(diff < 0,
                                 np.exp(-diff / 13) - 1,
                                 np.exp(diff  / 10) - 1)))

rmse_avg  = np.sqrt(mean_squared_error(true_rul, mean_pred))
rmse_last = np.sqrt(mean_squared_error(true_rul, last_pred))
s_avg     = compute_s_score(true_rul, mean_pred)
s_last    = compute_s_score(true_rul, last_pred)

print("\n" + "="*52)
print("   KEY METRICS  (CMAPSS FD001 — Piecewise LSTM)")
print("="*52)
print(f"  RMSE  (averaged {NUM_TEST_WINDOWS} windows) : {rmse_avg:.4f}")
print(f"  RMSE  (last window only)      : {rmse_last:.4f}")
print(f"  S-score (averaged windows)    : {s_avg:.4f}")
print(f"  S-score (last window only)    : {s_last:.4f}")
print("="*52)


# ─────────────────────────────────────────────────────────────
# 6. FIGURES
# ─────────────────────────────────────────────────────────────

# ── Figure 1: True vs Predicted RUL (averaged) ───────────────
fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(true_rul,  label="True RUL",      color="red",  linewidth=1.5)
ax1.plot(mean_pred, label="Predicted RUL", color="blue", linewidth=1.5, alpha=0.8)
ax1.set_xlabel("Engine Index")
ax1.set_ylabel("RUL (cycles)")
ax1.set_title(f"Figure 1 – True vs Predicted RUL (avg of {NUM_TEST_WINDOWS} windows)\n"
              f"LSTM Piecewise Model on CMAPSS FD001  |  RMSE = {rmse_avg:.2f}")
ax1.legend()
ax1.grid(True, alpha=0.4)
fig1.tight_layout()
fig1.savefig("figures/fig1_true_vs_pred_rul_avg.png", dpi=150)
plt.show()
print("Saved: figures/fig1_true_vs_pred_rul_avg.png")

# ── Figure 2: True vs Predicted RUL (last window only) ───────
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(true_rul,  label="True RUL",      color="red",  linewidth=1.5)
ax2.plot(last_pred, label="Predicted RUL", color="blue", linewidth=1.5, alpha=0.8)
ax2.set_xlabel("Engine Index")
ax2.set_ylabel("RUL (cycles)")
ax2.set_title(f"Figure 2 – True vs Predicted RUL (last window only)\n"
              f"LSTM Piecewise Model on CMAPSS FD001  |  RMSE = {rmse_last:.2f}")
ax2.legend()
ax2.grid(True, alpha=0.4)
fig2.tight_layout()
fig2.savefig("figures/fig2_true_vs_pred_rul_last.png", dpi=150)
plt.show()
print("Saved: figures/fig2_true_vs_pred_rul_last.png")

# ── Figure 3: Prediction Error Distribution ───────────────────
errors = last_pred - true_rul
fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.hist(errors, bins=20, edgecolor="black", color="steelblue", alpha=0.8)
ax3.axvline(0, color="red", linestyle="--", label="Zero error")
ax3.set_xlabel("Prediction Error  (Predicted − True RUL)")
ax3.set_ylabel("Frequency")
ax3.set_title("Figure 3 – Prediction Error Distribution\n"
              "LSTM Piecewise Model on CMAPSS FD001")
ax3.legend()
ax3.grid(True, alpha=0.4)
fig3.tight_layout()
fig3.savefig("figures/fig3_error_distribution.png", dpi=150)
plt.show()
print("Saved: figures/fig3_error_distribution.png")

# ── Figure 4: Scatter – Predicted vs True RUL ─────────────────
fig4, ax4 = plt.subplots(figsize=(7, 7))
ax4.scatter(true_rul, last_pred, alpha=0.7, edgecolors="k", linewidths=0.4)
lim = max(true_rul.max(), last_pred.max())
ax4.plot([0, lim], [0, lim], "r--", label="Perfect prediction")
ax4.set_xlabel("True RUL (cycles)")
ax4.set_ylabel("Predicted RUL (cycles)")
ax4.set_title("Figure 4 – Predicted vs True RUL\n"
              "LSTM Piecewise Model on CMAPSS FD001")
ax4.legend()
ax4.grid(True, alpha=0.4)
fig4.tight_layout()
fig4.savefig("figures/fig4_scatter_pred_vs_true.png", dpi=150)
plt.show()
print("Saved: figures/fig4_scatter_pred_vs_true.png")

# ── Results table ─────────────────────────────────────────────
results_df = pd.DataFrame({
    "unit":          np.arange(1, len(true_rul) + 1),
    "true_rul":      true_rul.round(1),
    "predicted_rul": last_pred.round(1),
    "error":         errors.round(1),
})
print("\nPer-engine results (first 10 rows):")
print(results_df.head(10).to_string(index=False))
results_df.to_csv("figures/per_engine_results.csv", index=False)
print("\nFull table saved: figures/per_engine_results.csv")

print("\n✓ All figures and metrics reproduced successfully.")
