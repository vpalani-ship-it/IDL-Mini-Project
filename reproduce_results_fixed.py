"""
reproduce_results.py
====================
Reproduces key figures and metrics for:
  Hybrid CNN-Attention-LSTM for RUL Estimation on NASA CMAPSS FD001

Usage:
    python reproduce_results.py

Requirements:
    pip install torch numpy pandas scikit-learn matplotlib gdown

Expected repo layout:
    reproduce_results.py
    cnn_attention_lstm_fd001.pt
    CMAPSSData/
        train_FD001.txt
        test_FD001.txt
        RUL_FD001.txt

What this script does, without retraining:
    1. Downloads the trained checkpoint from Google Drive if not already present.
    2. Loads CMAPSS FD001.
    3. Preprocesses train/test data consistently.
    4. Loads saved CNN-Attention-LSTM weights.
    5. Runs final-window inference on the 100 test engines.
    6. Prints RMSE, MAE, R^2, and NASA S-score.
    7. Saves key figures and a predictions table to ./figures/.
"""

# ─────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────────────────────
GDRIVE_FILE_ID  = "1Gv4i5e-0vWMdjwwxNgwFDsT4XsC_A2Br"
CHECKPOINT_PATH = "cnn_attention_lstm_fd001.pt"
DATA_DIR        = "CMAPSSData"
FIGURE_DIR      = "figures"

# If your checkpoint was trained with a different window size, set it here.
# The script will use checkpoint["window_size"] if it exists.
DEFAULT_WINDOW_SIZE = 30

# ─────────────────────────────────────────────────────────────
# 1. IMPORTS AND SETUP
# ─────────────────────────────────────────────────────────────
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 313
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

os.makedirs(FIGURE_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 2. CHECKPOINT DOWNLOAD
# ─────────────────────────────────────────────────────────────
def download_checkpoint():
    """Download model checkpoint if it is not already present."""
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint already exists at '{CHECKPOINT_PATH}'. Skipping download.")
        return

    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "gdown is required for auto-download. Install it with:\n"
            "    pip install gdown\n"
            "Or manually download the .pt checkpoint and place it next to this script."
        ) from exc

    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    print(f"Downloading checkpoint from Google Drive -> '{CHECKPOINT_PATH}' ...")
    gdown.download(url, CHECKPOINT_PATH, quiet=False)
    print("Download complete.")


# ─────────────────────────────────────────────────────────────
# 3. MODEL DEFINITION — must match training notebook
# ─────────────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_rate=2):
        super().__init__()
        hidden = max(1, channels // reduction_rate)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(attention))


class CNNAttentionLSTM(nn.Module):
    def __init__(self, window_size, num_features, lstm_hidden=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=(10, 3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), padding=0, ceil_mode=True),

            nn.Conv2d(10, 10, kernel_size=(3, 3), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), padding=0, ceil_mode=True),

            nn.Conv2d(10, 10, kernel_size=(5, 5), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), padding=0, ceil_mode=True),
        )
        self.channel_attention = ChannelAttention(channels=10, reduction_rate=2)
        self.spatial_attention = SpatialAttention(kernel_size=3)

        # Infer LSTM input size after CNN + attention.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, window_size, num_features)
            out = self.spatial_attention(self.channel_attention(self.cnn(dummy)))
            _, channels, time_reduced, features_reduced = out.shape
            self.lstm_input_size = channels * features_reduced
            self.lstm_time_steps = time_reduced

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden,
            batch_first=True,
        )
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        x = self.cnn(x)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        # CNN output shape: (batch, channels, time, features)
        # LSTM input shape: (batch, time, channels * features)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size(0), x.size(1), -1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


# ─────────────────────────────────────────────────────────────
# 4. DATA LOADING AND PREPROCESSING
# ─────────────────────────────────────────────────────────────
def load_and_preprocess():
    """Load CMAPSS FD001 and reproduce preprocessing used for final-window evaluation."""
    train_path = os.path.join(DATA_DIR, "train_FD001.txt")
    test_path = os.path.join(DATA_DIR, "test_FD001.txt")
    rul_path = os.path.join(DATA_DIR, "RUL_FD001.txt")

    for path in [train_path, test_path, rul_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Could not find '{path}'. Expected files under DATA_DIR='{DATA_DIR}'."
            )

    # CMAPSS FD001 has 26 columns:
    # unit + time + 3 operational settings + 21 sensors.
    column_names = ["unit", "time", "c1", "c2", "c3"] + [f"s{i}" for i in range(1, 22)]

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=column_names)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=column_names)
    rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["rul"])
    rul_df["unit"] = np.arange(1, len(rul_df) + 1)

    # Cast features to float.
    feature_like_cols = [c for c in train_df.columns if c.startswith("c") or c.startswith("s")]
    train_df[feature_like_cols] = train_df[feature_like_cols].astype(float)
    test_df[feature_like_cols] = test_df[feature_like_cols].astype(float)

    # Drop low-information columns commonly removed for FD001.
    non_informative = ["c3", "s1", "s5", "s6", "s10", "s16", "s18", "s19"]
    train_df = train_df.drop(columns=non_informative, errors="raise")
    test_df = test_df.drop(columns=non_informative, errors="raise")

    feature_columns = [
        c for c in train_df.columns
        if c.startswith("c") or c.startswith("s")
    ]

    # Normalize features. Fit on train only, then transform train and test.
    scaler = MinMaxScaler()
    train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
    test_df[feature_columns] = scaler.transform(test_df[feature_columns])

    # Training RUL: max cycle for each engine minus current cycle.
    max_cycles = train_df.groupby("unit")["time"].max().reset_index()
    max_cycles.columns = ["unit", "max_time"]
    train_df = train_df.merge(max_cycles, on="unit", how="left")
    train_df["rul"] = train_df["max_time"] - train_df["time"]
    train_df = train_df.drop(columns=["max_time"])

    # Test RUL: remaining cycles inside observed test segment + official final RUL.
    test_max_cycles = test_df.groupby("unit")["time"].max().reset_index()
    test_max_cycles.columns = ["unit", "max_time"]
    test_df = test_df.merge(test_max_cycles, on="unit", how="left")
    test_df = test_df.merge(rul_df, on="unit", how="left", suffixes=("", "_final"))
    test_df["rul"] = (test_df["max_time"] - test_df["time"]) + test_df["rul"]
    test_df = test_df.drop(columns=["max_time"])

    # Piecewise-linear RUL cap.
    RUL_CAP = 125
    TIME_WINDOW = 50
    train_df["rul"] = train_df["rul"].clip(upper=RUL_CAP)
    test_df["rul"] = test_df["rul"].clip(upper=RUL_CAP)

    # Optional label kept for compatibility with notebook outputs.
    train_df["label"] = (train_df["rul"] > TIME_WINDOW).astype(int)
    test_df["label"] = (test_df["rul"] > TIME_WINDOW).astype(int)

    return train_df, test_df, feature_columns


def create_test_windows(df, feature_columns, window_size=30):
    """Create one final window per test engine — standard CMAPSS benchmark."""
    X_reg, y_reg = [], []

    for engine_id in sorted(df["unit"].unique()):
        engine_df = df[df["unit"] == engine_id].sort_values("time")
        n_rows = len(engine_df)

        if n_rows >= window_size:
            final_window = engine_df.iloc[-window_size:]
        else:
            pad_len = window_size - n_rows
            pad = pd.concat([engine_df.iloc[[0]]] * pad_len, ignore_index=True)
            final_window = pd.concat([pad, engine_df], ignore_index=True)

        X_reg.append(final_window[feature_columns].values)
        y_reg.append(engine_df["rul"].values[-1])

    return np.asarray(X_reg, dtype=np.float32), np.asarray(y_reg, dtype=np.float32)


def to_torch_4d(X):
    """Convert from (N, T, F) to PyTorch Conv2D input (N, 1, T, F)."""
    return torch.tensor(np.asarray(X, dtype=np.float32)).unsqueeze(1)


# ─────────────────────────────────────────────────────────────
# 5. LOAD DATA, CHECKPOINT, AND RUN INFERENCE
# ─────────────────────────────────────────────────────────────
def load_checkpoint(path):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint, checkpoint["model_state_dict"]
    # fallback for checkpoints saved directly as model.state_dict()
    return {}, checkpoint


def predict(model, loader, device):
    model.eval()
    preds_list, y_list = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch.to(device)).cpu().numpy().reshape(-1)
            preds_list.append(preds)
            y_list.append(y_batch.numpy().reshape(-1))

    return np.concatenate(y_list), np.concatenate(preds_list)


# ─────────────────────────────────────────────────────────────
# 6. METRICS AND PLOTS
# ─────────────────────────────────────────────────────────────
def compute_s_score(y_true, y_pred):
    d = np.asarray(y_pred).reshape(-1) - np.asarray(y_true).reshape(-1)
    return float(
        np.sum(
            np.where(
                d < 0,
                np.exp(-d / 13.0) - 1.0,
                np.exp(d / 10.0) - 1.0,
            )
        )
    )


def save_figures(y_true, y_pred):
    errors = y_pred - y_true
    engine_ids = np.arange(1, len(y_true) + 1)

    # Figure 1: Predicted vs True RUL
    fig1, ax1 = plt.subplots(figsize=(7, 7))
    ax1.scatter(y_true, y_pred, alpha=0.7, edgecolors="k", linewidths=0.4)
    line_max = float(max(np.max(y_true), np.max(y_pred)))
    ax1.plot([0, line_max], [0, line_max], "r--", label="Perfect prediction")
    ax1.set_xlabel("True RUL (cycles)")
    ax1.set_ylabel("Predicted RUL (cycles)")
    ax1.set_title("Predicted vs True RUL\nCNN-Attention-LSTM on CMAPSS FD001")
    ax1.legend()
    ax1.grid(True, alpha=0.4)
    fig1.tight_layout()
    fig1.savefig(os.path.join(FIGURE_DIR, "fig1_pred_vs_true_rul.png"), dpi=150)
    plt.close(fig1)

    # Figure 2: Prediction Error Distribution
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(errors, bins=20, edgecolor="black", alpha=0.8)
    ax2.axvline(0, color="red", linestyle="--", label="Zero error")
    ax2.set_xlabel("Prediction Error (Predicted − True RUL)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Prediction Error Distribution\nCNN-Attention-LSTM on CMAPSS FD001")
    ax2.legend()
    ax2.grid(True, alpha=0.4)
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIGURE_DIR, "fig2_error_distribution.png"), dpi=150)
    plt.close(fig2)

    # Figure 3: Per-engine Error Bar Chart
    fig3, ax3 = plt.subplots(figsize=(14, 5))
    ax3.bar(engine_ids, errors, edgecolor="k", linewidth=0.3)
    ax3.axhline(0, color="black", linewidth=1)
    ax3.set_xlabel("Test Engine ID")
    ax3.set_ylabel("Prediction Error (Predicted − True RUL)")
    ax3.set_title("Per-Engine Prediction Error\nCNN-Attention-LSTM on CMAPSS FD001")
    ax3.grid(True, axis="y", alpha=0.4)
    fig3.tight_layout()
    fig3.savefig(os.path.join(FIGURE_DIR, "fig3_per_engine_error.png"), dpi=150)
    plt.close(fig3)

    results_df = pd.DataFrame({
        "unit": engine_ids,
        "true_rul": np.round(y_true, 1),
        "predicted_rul": np.round(y_pred, 1),
        "error": np.round(errors, 1),
    })
    results_df.to_csv(os.path.join(FIGURE_DIR, "per_engine_results.csv"), index=False)

    print(f"Saved figures and table to ./{FIGURE_DIR}/")
    print(results_df.head(10).to_string(index=False))


# ─────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────
def main():
    download_checkpoint()

    print("\n── Loading & preprocessing CMAPSS FD001 data ──")
    train_df, test_df, feature_columns = load_and_preprocess()

    checkpoint, state_dict = load_checkpoint(CHECKPOINT_PATH)
    window_size = int(checkpoint.get("window_size", DEFAULT_WINDOW_SIZE))
    num_features = int(checkpoint.get("num_features", len(feature_columns)))

    X_reg_test, y_reg_test = create_test_windows(test_df, feature_columns, window_size)
    X_test_t = to_torch_4d(X_reg_test)
    y_test_t = torch.tensor(y_reg_test, dtype=torch.float32).view(-1, 1)

    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=64,
        shuffle=False,
    )

    print(f"Test set tensor: {tuple(X_test_t.shape)} | {len(y_reg_test)} engines")
    print(f"Using window_size={window_size}, num_features={num_features}")

    if X_test_t.shape[-1] != num_features:
        raise ValueError(
            f"Feature-count mismatch: checkpoint expects {num_features}, "
            f"but preprocessing produced {X_test_t.shape[-1]}. "
            "Check that feature dropping/normalization matches the training notebook."
        )

    print(f"\n── Loading checkpoint: {CHECKPOINT_PATH} ──")
    model = CNNAttentionLSTM(
        window_size=window_size,
        num_features=num_features,
        lstm_hidden=128,
    ).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully.")
    print(f"  LSTM input size : {model.lstm_input_size}")
    print(f"  LSTM time steps : {model.lstm_time_steps}")

    y_true, y_pred = predict(model, test_loader, DEVICE)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    s_score = compute_s_score(y_true, y_pred)

    print("\n" + "=" * 45)
    print("   KEY METRICS  (standard CMAPSS FD001 test)")
    print("=" * 45)
    print(f"  MAE          : {mae:.3f}")
    print(f"  MSE          : {mse:.3f}")
    print(f"  RMSE         : {rmse:.3f}")
    print(f"  R²           : {r2:.3f}")
    print(f"  NASA S-score : {s_score:.3f}")
    print("=" * 45)

    if checkpoint and "metrics" in checkpoint:
        print("\nSaved training-time metrics for reference:")
        for key, value in checkpoint["metrics"].items():
            try:
                print(f"  {key:12s}: {float(value):.3f}")
            except Exception:
                print(f"  {key:12s}: {value}")

    save_figures(y_true, y_pred)
    print("\n✓ All figures and metrics reproduced successfully.")


if __name__ == "__main__":
    main()
