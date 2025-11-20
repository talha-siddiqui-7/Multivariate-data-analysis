#!/usr/bin/env python3
"""
Synthetic example where LSTM should clearly outperform:
- Persistence baseline (y_hat(t+1) = y(t))
- Linear Regression on windowed inputs

Data:
- Time resolution: 15 min
- Features: feat1, feat2, feat3 (nonlinear, oscillatory)
- Target: SUPPLY RH = nonlinear function of lagged features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

# ============================================================
# ----------------- 1. SYNTHETIC DATA ------------------------
# ============================================================

N_STEPS = 5000           # length of time series
DT_HOURS = 0.25          # 15 min resolution
LOOKBACK = 32            # same as before (~8 h window)
TEST_FRACTION = 0.15
VAL_FRACTION  = 0.15
BATCH_SIZE    = 64
EPOCHS        = 50

np.random.seed(42)

# Time axis (in "simulation hours", like your real data)
time = np.arange(N_STEPS) * DT_HOURS

# Base oscillatory signals (smooth but nonlinear in time)
f1 = np.sin(2 * np.pi * time / 50.0)      # slower oscillation
f2 = np.sin(2 * np.pi * time / 20.0 + 1)  # faster, phase shifted
f3 = np.cos(2 * np.pi * time / 35.0)      # another component

# Add some noise
f1 += 0.1 * np.random.randn(N_STEPS)
f2 += 0.1 * np.random.randn(N_STEPS)
f3 += 0.1 * np.random.randn(N_STEPS)

# Create lagged versions (nonlinear temporal dependence)
f1_l1 = np.roll(f1, 1)
f2_l3 = np.roll(f2, 3)
f1_l2 = np.roll(f1, 2)
f2_l5 = np.roll(f2, 5)

# Strongly nonlinear target with lagged interactions
# (linear regression cannot represent sin/cos of products well)
y = (
    45
    + 10 * np.sin(f1_l1 * f2_l3)          # nonlinear product with lag
    + 5 * np.cos(f1_l2 + f2_l5)           # nonlinear sum with other lag
    + 2 * f3                              # some linear contribution
    + 0.5 * np.random.randn(N_STEPS)      # small noise
)

# Clip to a plausible RH range (e.g. 30–60%)
y = np.clip(y, 30, 60)

# Fix the first few points where roll wraps around (not important)
y[:10] = y[10]

# Build DataFrame similar to your real one
df = pd.DataFrame({
    "Time": time,
    "feat1": f1,
    "feat2": f2,
    "feat3": f3,
    "SUPPLY RH": y,
})

print(df.head())
print("Synthetic data shape:", df.shape)

# ============================================================
# ----------- 2. PREPARE DATA (same as your code) ------------
# ============================================================

TARGET_COL = "SUPPLY RH"
TIME_COL   = "Time"

# Separate features and target
feature_cols = [c for c in df.columns if c not in [TARGET_COL]]
X_raw = df[feature_cols].values
y_raw = df[[TARGET_COL]].values         # shape (n, 1)
time_raw = df[TIME_COL].values

# Scale features and target
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_scaled = x_scaler.fit_transform(X_raw)
y_scaled = y_scaler.fit_transform(y_raw)

print("Feature shape (scaled):", X_scaled.shape)
print("Target shape (scaled) :", y_scaled.shape)

# ---- create windows ----
def create_windows(X, y, time, lookback):
    X_seq, y_seq, time_seq = [], [], []
    for t in range(lookback, len(X)):
        X_seq.append(X[t - lookback:t, :])
        y_seq.append(y[t, 0])
        time_seq.append(time[t])
    return np.array(X_seq), np.array(y_seq), np.array(time_seq)

X_seq, y_seq, time_seq = create_windows(X_scaled, y_scaled, time_raw, LOOKBACK)
print("X_seq:", X_seq.shape, "y_seq:", y_seq.shape)

n_samples = len(X_seq)
test_size = int(TEST_FRACTION * n_samples)
val_size  = int(VAL_FRACTION * n_samples)
train_size = n_samples - test_size - val_size

X_train = X_seq[:train_size]
y_train = y_seq[:train_size]

X_val   = X_seq[train_size:train_size + val_size]
y_val   = y_seq[train_size:train_size + val_size]

X_test  = X_seq[train_size + val_size:]
y_test  = y_seq[train_size + val_size:]

time_train = time_seq[:train_size]
time_val   = time_seq[train_size:train_size + val_size]
time_test  = time_seq[train_size + val_size:]

print("Train:", X_train.shape[0], "Val:", X_val.shape[0], "Test:", X_test.shape[0])

# ============================================================
# ----------------- 3. LSTM MODEL (same arch) ----------------
# ============================================================

n_features = X_train.shape[2]

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(LOOKBACK, n_features)))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)

# ============================================================
# ----------------- 4. EVALUATION + BASELINES ----------------
# ============================================================

def invert_target_scaling(y_scaled_1d):
    y_scaled_2d = y_scaled_1d.reshape(-1, 1)
    return y_scaler.inverse_transform(y_scaled_2d).ravel()

def print_metrics(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    print(f"{name} -> RMSE: {rmse:.3f}, MAE: {mae:.3f}")
    return rmse, mae

# ----- LSTM predictions -----
y_train_pred = model.predict(X_train).ravel()
y_val_pred   = model.predict(X_val).ravel()
y_test_pred  = model.predict(X_test).ravel()

y_train_true = invert_target_scaling(y_train)
y_val_true   = invert_target_scaling(y_val)
y_test_true  = invert_target_scaling(y_test)

y_train_pred = invert_target_scaling(y_train_pred)
y_val_pred   = invert_target_scaling(y_val_pred)
y_test_pred  = invert_target_scaling(y_test_pred)

print("\n=== LSTM performance ===")
rmse_train_lstm, mae_train_lstm = print_metrics("Train (LSTM)", y_train_true, y_train_pred)
rmse_val_lstm,   mae_val_lstm   = print_metrics("Val   (LSTM)", y_val_true,   y_val_pred)
rmse_test_lstm,  mae_test_lstm  = print_metrics("Test  (LSTM)", y_test_true,  y_test_pred)

# ----- Baseline 1: Persistence -----
baseline_seq = np.zeros_like(y_seq)
baseline_seq[0] = y_seq[0]
baseline_seq[1:] = y_seq[:-1]

baseline_train = baseline_seq[:train_size]
baseline_val   = baseline_seq[train_size:train_size + val_size]
baseline_test  = baseline_seq[train_size + val_size:]

y_train_pred_pers = invert_target_scaling(baseline_train)
y_val_pred_pers   = invert_target_scaling(baseline_val)
y_test_pred_pers  = invert_target_scaling(baseline_test)

print("\n=== Persistence baseline (y_hat(t+1) = y(t)) ===")
rmse_train_pers, mae_train_pers = print_metrics("Train (Pers)", y_train_true, y_train_pred_pers)
rmse_val_pers,   mae_val_pers   = print_metrics("Val   (Pers)", y_val_true,   y_val_pred_pers)
rmse_test_pers,  mae_test_pers  = print_metrics("Test  (Pers)", y_test_true,  y_test_pred_pers)

# ----- Baseline 2: Linear Regression on windowed inputs -----
X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_val_flat   = X_val.reshape((X_val.shape[0], -1))
X_test_flat  = X_test.reshape((X_test.shape[0], -1))

linreg = LinearRegression()
linreg.fit(X_train_flat, y_train)   # y_train is scaled

y_train_pred_lr_scaled = linreg.predict(X_train_flat)
y_val_pred_lr_scaled   = linreg.predict(X_val_flat)
y_test_pred_lr_scaled  = linreg.predict(X_test_flat)

y_train_pred_lr = invert_target_scaling(y_train_pred_lr_scaled)
y_val_pred_lr   = invert_target_scaling(y_val_pred_lr_scaled)
y_test_pred_lr  = invert_target_scaling(y_test_pred_lr_scaled)

print("\n=== Linear Regression baseline ===")
rmse_train_lr, mae_train_lr = print_metrics("Train (LinReg)", y_train_true, y_train_pred_lr)
rmse_val_lr,   mae_val_lr   = print_metrics("Val   (LinReg)", y_val_true,   y_val_pred_lr)
rmse_test_lr,  mae_test_lr  = print_metrics("Test  (LinReg)", y_test_true,  y_test_pred_lr)

# ----- Comparison table -----
print("\n================ COMPARISON (Test set) ================")
print("Model        | RMSE [%] | MAE [%]")
print("-------------+----------+--------")
print(f"Persistence  | {rmse_test_pers:7.3f} | {mae_test_pers:6.3f}")
print(f"LinearReg    | {rmse_test_lr:7.3f} | {mae_test_lr:6.3f}")
print(f"LSTM         | {rmse_test_lstm:7.3f} | {mae_test_lstm:6.3f}")
print("========================================================\n")

# ============================================================
# ------------------------- PLOTS ----------------------------
# ============================================================

# Training curves
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("Training vs Validation Loss (Synthetic SUPPLY RH LSTM)")
plt.legend()
plt.tight_layout()
plt.show()

# Time-series: last N_show points
N_show = 400
if len(y_test_true) < N_show:
    N_show = len(y_test_true)

plt.figure(figsize=(10, 4))
plt.plot(time_test[-N_show:], y_test_true[-N_show:], label="Actual SUPPLY RH")
plt.plot(time_test[-N_show:], y_test_pred[-N_show:], label="Predicted SUPPLY RH (LSTM)", linestyle="--")
plt.xlabel("Time [h]")
plt.ylabel("SUPPLY RH [%]")
plt.title(f"Synthetic test set: Actual vs Predicted SUPPLY RH (last {N_show} points)")
plt.legend()
plt.tight_layout()
plt.show()

# Scatter plot
plt.figure(figsize=(5, 5))
plt.scatter(y_test_true, y_test_pred, alpha=0.4, edgecolor="k")
mn = min(y_test_true.min(), y_test_pred.min())
mx = max(y_test_true.max(), y_test_pred.max())
plt.plot([mn, mx], [mn, mx], "r--", label="1:1 line")
plt.xlabel("Actual SUPPLY RH [%]")
plt.ylabel("Predicted SUPPLY RH [%]")
plt.title("Synthetic test set: Actual vs Predicted SUPPLY RH (LSTM)")
plt.legend()
plt.tight_layout()
plt.show()
