#!/usr/bin/env python3
"""
EM-PCA analysis of swimming pool dataset.

Creates:
1) Scree plot + cumulative variance
2) Loadings bar plot (PC1 & PC2)
3) Time series of PC1 scores
4) PC1 vs PC2 scatter (coloured by operating proxy)
5) Reconstruction error comparison (Mean-PCA vs EM-PCA)
6) Cross-validation of #components (EM-PCA)
7) Scaling sensitivity plot (standardised vs raw PCA)

Run in PyCharm: just hit Run; figures will pop up.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Tuple, List

# ============================================================
# ------------------- USER SETTINGS --------------------------
# ============================================================

# <<< CHANGE THIS PATH TO YOUR CSV LOCATION >>>
FILE_PATH = r"M:\PhD\03 Experiments\17-09-2025_16-10-2025_sensor_vs_AHU_data.csv"

# Variables to include in PCA (only those that exist will be used)
SELECTED_COLUMNS = [
    "Extract_air_Temp_sensor",
    "Extract_air_RH_sensor",
    "Supply_air_temp_sensor",
    "Supply_air_RH_sensor",
    "Outdoor_Temp",
    "Outdoor_RH",
    "Pool_OF",
    "Pool_water_temp_AHU",
    "Setpoint Pool water temperature",
    "Vdot_sup_m3s",
    "Vdot_ext_m3s",
    "Fresh air damper",
    "w_sup",
    "w_ext",
    "w_out",
    "inf_moist_kg_s",
    "inf_dry_from_water_kg_s"
]

N_COMPONENTS_MAIN = 4       # how many PCs to keep in the main model
EM_MAX_ITER = 40
EM_TOL = 1e-4
RANDOM_STATE = 0

# For reconstruction-error validation
CV_FRAC_MASK = 0.05         # fraction of observed entries to mask
CV_REPEATS = 3
CV_MAX_COMPONENTS = 8

# ============================================================
# ---------------------- HELPERS -----------------------------
# ============================================================

def find_time_column(df: pd.DataFrame) -> str:
    """Try to guess the time column name."""
    candidates = ["datetime", "Datetime", "time", "Time", "timestamp", "Timestamp"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("No datetime/time column found. Please adjust 'find_time_column'.")


def standardise_matrix(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardise columns: (x - mean) / std, ignoring NaNs."""
    col_means = np.nanmean(X, axis=0)
    col_stds = np.nanstd(X, axis=0, ddof=1)
    # Avoid division by zero
    col_stds[col_stds == 0] = 1.0
    X_std = (X - col_means) / col_stds
    return X_std, col_means, col_stds


def mean_impute(X: np.ndarray) -> np.ndarray:
    """Impute NaNs with column means."""
    X_imp = X.copy()
    col_means = np.nanmean(X_imp, axis=0)
    inds = np.where(np.isnan(X_imp))
    X_imp[inds] = np.take(col_means, inds[1])
    return X_imp


def run_em_pca(
    X_std: np.ndarray,
    n_components: int,
    max_iter: int = 40,
    tol: float = 1e-4,
    random_state: int = 0
) -> Tuple[PCA, np.ndarray, np.ndarray]:
    """
    EM-PCA on a standardised data matrix with NaNs.

    Returns:
        pca      : fitted sklearn PCA object
        scores   : score matrix T (n_samples, n_components)
        X_filled : reconstructed matrix in standardised space
    """
    X = X_std.copy()
    missing = np.isnan(X)

    # Initial imputation with column means (which are ~0 after standardisation)
    col_means = np.nanmean(X, axis=0)
    X[missing] = np.take(col_means, np.where(missing)[1])

    for it in range(max_iter):
        pca = PCA(n_components=n_components, random_state=random_state)
        scores = pca.fit_transform(X)
        X_recon = pca.inverse_transform(scores)

        # Change only on missing entries
        diff_missing = X_recon[missing] - X[missing]
        max_change = np.nanmax(np.abs(diff_missing))

        X[missing] = X_recon[missing]

        print(f"[EM-PCA] Iter {it + 1:02d}: max change on missing entries = {max_change:.6f}")

        if max_change < tol:
            print("[EM-PCA] Converged.")
            break

    return pca, scores, X


def reconstruction_rmse(true_vals: np.ndarray, pred_vals: np.ndarray) -> float:
    """Root-mean-square error between two 1D arrays."""
    mse = np.mean((true_vals - pred_vals) ** 2)
    return float(np.sqrt(mse))


# ============================================================
# -------------------- LOAD & PREP DATA ----------------------
# ============================================================

print("Loading data...")
df = pd.read_csv(FILE_PATH)

# Find and parse time column
time_col = find_time_column(df)
df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
df = df.sort_values(time_col).reset_index(drop=True)

# Keep only columns that actually exist
used_cols: List[str] = [c for c in SELECTED_COLUMNS if c in df.columns]
missing_cols = [c for c in SELECTED_COLUMNS if c not in df.columns]

print(f"Using {len(used_cols)} variables for PCA:")
for c in used_cols:
    print(f"  - {c}")

if missing_cols:
    print("\nWARNING: These columns were not found and will be ignored:")
    for c in missing_cols:
        print(f"  - {c}")

# Fix decimal comma problem and force numeric
for col in used_cols:
    df[col] = (
        df[col]
        .astype(str)                         # ensure string
        .str.replace(",", ".", regex=False)  # "31,3" -> "31.3"
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")  # non-numeric -> NaN

# Now safely convert to numpy
X_raw = df[used_cols].to_numpy(dtype=float)
time_values = df[time_col].to_numpy()

# ============================================================
# ------------- STANDARDISE & CREATE MASK --------------------
# ============================================================

X_std, col_means, col_stds = standardise_matrix(X_raw)
missing_mask = np.isnan(X_std)

# ============================================================
# ---------------------- MAIN EM-PCA -------------------------
# ============================================================

print("\nRunning EM-PCA on full dataset...")
pca_em, scores_em, X_em_filled = run_em_pca(
    X_std,
    n_components=N_COMPONENTS_MAIN,
    max_iter=EM_MAX_ITER,
    tol=EM_TOL,
    random_state=RANDOM_STATE,
)

explained_var_ratio = pca_em.explained_variance_ratio_
cum_explained = np.cumsum(explained_var_ratio)

# ============================================================
# -------------------------- PLOTS ---------------------------
# ============================================================

# 1) Scree plot + cumulative variance
plt.figure(figsize=(8, 5))
components = np.arange(1, N_COMPONENTS_MAIN + 1)
plt.bar(components, explained_var_ratio * 100, alpha=0.7)
plt.plot(components, cum_explained * 100, marker="o")
plt.xlabel("Principal Component")
plt.ylabel("Explained variance [%]")
plt.title("Scree plot & cumulative explained variance (EM-PCA)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

# 2) Loadings bar plot for PC1 & PC2
loadings = pca_em.components_  # shape (n_components, n_features)
feature_names = np.array(used_cols)

plt.figure(figsize=(10, 6))
x = np.arange(len(feature_names))
width = 0.35

plt.bar(x - width / 2, loadings[0, :], width, label="PC1")
plt.bar(x + width / 2, loadings[1, :], width, label="PC2")
plt.xticks(x, feature_names, rotation=45, ha="right")
plt.ylabel("Loading weight")
plt.title("Loadings for PC1 and PC2 (standardised variables)")
plt.legend()
plt.grid(True, axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

# Optional: simple biplot (PC1 vs PC2 loadings)
plt.figure(figsize=(7, 7))
ax = plt.gca()

plt.axhline(0, color="grey", linewidth=0.5)
plt.axvline(0, color="grey", linewidth=0.5)

for i, name in enumerate(feature_names):
    plt.arrow(0, 0, loadings[0, i], loadings[1, i],
              head_width=0.02, length_includes_head=True)
    plt.text(loadings[0, i] * 1.05, loadings[1, i] * 1.05, name,
             ha="center", va="center")

plt.xlabel("PC1 loading")
plt.ylabel("PC2 loading")
plt.title("Biplot of variable loadings (PC1 vs PC2)")

# ------------------------------------------------------------
# Add physical meaning of + / - directions for PC1 and PC2
# ------------------------------------------------------------

# PC1 interpretation (x-axis)
plt.text(0.95, 0.02,
         "+PC1 → lower humidity / stronger ventilation\n"
         "-PC1 → high humidity / evaporation",
         transform=ax.transAxes,
         fontsize=9, ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

# PC2 interpretation (y-axis)
plt.text(0.02, 0.95,
         "+PC2 → high ventilation / mechanical drying\n"
         "-PC2 → low ventilation / night mode",
         transform=ax.transAxes,
         fontsize=9, ha="left", va="top",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()


# 3) Time series of PC1 scores
plt.figure(figsize=(10, 5))
plt.plot(time_values, scores_em[:, 0], linewidth=0.8)
plt.xlabel("Time")
plt.ylabel("PC1 score")
plt.title("PC1 score over time (EM-PCA)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

# 4) Improved PC1 vs PC2 scatter, coloured by operating proxy

# Determine colouring variable for operating regimes
color_var = None
color_label = None

if "Fresh air damper" in df.columns:
    color_var = df["Fresh air damper"].to_numpy(dtype=float)
    color_label = "Fresh air damper"
elif "Setpoint Pool water temperature" in df.columns:
    color_var = df["Setpoint Pool water temperature"].to_numpy(dtype=float)
    color_label = "Pool water setpoint"
else:
    color_var = None

plt.figure(figsize=(8, 7))
if color_var is not None:
    sc = plt.scatter(scores_em[:, 0], scores_em[:, 1],
                     c=color_var, s=8, alpha=0.6, cmap="viridis")
    cbar = plt.colorbar(sc)
    cbar.set_label(color_label)
else:
    plt.scatter(scores_em[:, 0], scores_em[:, 1], s=8, alpha=0.6)

plt.xlabel("PC1 score")
plt.ylabel("PC2 score")
plt.title("PC1 vs PC2 scores (operating modes)")

# ============================================================
# NEW PLOT: Average daily profile of PC1 with interpretation box
# ============================================================

pc1_df = pd.DataFrame({
    "time": time_values,
    "PC1": scores_em[:, 0]
})
pc1_df["hour"] = pc1_df["time"].dt.hour
pc1_df["minute"] = pc1_df["time"].dt.minute
pc1_df["tod"] = pc1_df["hour"] + pc1_df["minute"] / 60  # hours

avg_daily = pc1_df.groupby("tod")["PC1"].mean()
std_daily = pc1_df.groupby("tod")["PC1"].std()

plt.figure(figsize=(12, 6))
ax = plt.gca()

plt.plot(avg_daily.index, avg_daily.values, label="Average PC1 profile", linewidth=2)
plt.fill_between(avg_daily.index,
                 avg_daily.values - std_daily.values,
                 avg_daily.values + std_daily.values,
                 alpha=0.2, label="±1 std")

plt.xlabel("Time of day [hours]")
plt.ylabel("PC1 score")
plt.title("Average Daily Pattern of PC1 (Humidity/Evaporation Mode)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()

# ------------------------------------------------------------
# Add small interpretation text box on the y-axis
# ------------------------------------------------------------
plt.text(0.02, 0.90,
         "+PC1 → high humidity / high evaporation\n"
         "-PC1 → low humidity / drying / night mode",
         transform=ax.transAxes,
         fontsize=9, ha='left', va='top',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.7))

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Add physical interpretation annotations on axes
# ------------------------------------------------------------
# For PC1 (x-axis)
plt.text(0.95, 0.02,
         "+PC1 → lower humidity / strong ventilation\n-PC1 → high humidity / evaporation",
         transform=plt.gca().transAxes,
         fontsize=9, ha='right', va='bottom',
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

# For PC2 (y-axis)
plt.text(0.02, 0.95,
         "+PC2 → high ventilation / mechanical drying\n-PC2 → low ventilation / night mode",
         transform=plt.gca().transAxes,
         fontsize=9, ha='left', va='top',
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# ------ VALIDATION: MEAN-PCA vs EM-PCA RECONSTRUCTION -------
# ============================================================

print("\nValidation: reconstruction error (Mean-PCA vs EM-PCA)...")

# Only consider positions that were originally observed
X_std_orig = X_std.copy()
observed_mask = ~missing_mask
obs_indices = np.where(observed_mask)
n_total_obs = len(obs_indices[0])

rmse_mean_list = []
rmse_em_list = []

for rep in range(CV_REPEATS):
    print(f"  Repeat {rep + 1}/{CV_REPEATS}...")

    # Choose a subset of observed positions to mask for validation
    n_mask = int(CV_FRAC_MASK * n_total_obs)
    chosen = np.random.choice(n_total_obs, size=n_mask, replace=False)
    rows = obs_indices[0][chosen]
    cols = obs_indices[1][chosen]

    X_cv = X_std_orig.copy()
    X_cv[rows, cols] = np.nan

    # --- Mean-imputation PCA ---
    X_mean_imp = mean_impute(X_cv)
    pca_mean = PCA(n_components=N_COMPONENTS_MAIN, random_state=RANDOM_STATE)
    scores_mean = pca_mean.fit_transform(X_mean_imp)
    X_mean_recon = pca_mean.inverse_transform(scores_mean)

    # --- EM-PCA ---
    pca_cv_em, scores_cv_em, X_cv_em = run_em_pca(
        X_cv,
        n_components=N_COMPONENTS_MAIN,
        max_iter=EM_MAX_ITER,
        tol=EM_TOL,
        random_state=RANDOM_STATE + rep + 1,
    )

    true_vals = X_std_orig[rows, cols]
    pred_mean = X_mean_recon[rows, cols]
    pred_em = X_cv_em[rows, cols]

    rmse_mean = reconstruction_rmse(true_vals, pred_mean)
    rmse_em = reconstruction_rmse(true_vals, pred_em)

    rmse_mean_list.append(rmse_mean)
    rmse_em_list.append(rmse_em)

    print(f"    RMSE Mean-PCA = {rmse_mean:.4f}, EM-PCA = {rmse_em:.4f}")

rmse_mean_avg = float(np.mean(rmse_mean_list))
rmse_em_avg = float(np.mean(rmse_em_list))

# Bar plot of reconstruction RMSE
plt.figure(figsize=(6, 5))
methods = ["Mean-PCA", "EM-PCA"]
rmse_values = [rmse_mean_avg, rmse_em_avg]
plt.bar(methods, rmse_values, alpha=0.8)
plt.ylabel("Reconstruction RMSE (std units)")
plt.title("Reconstruction error comparison\n(masked observed entries)")
plt.grid(True, axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# --------- VALIDATION: CV vs #COMPONENTS (EM-PCA) -----------
# ============================================================

print("\nCross-validation of number of components (EM-PCA)...")

Ks = list(range(1, CV_MAX_COMPONENTS + 1))
cv_errors = []

for K in Ks:
    print(f"  K = {K}")
    rmse_list = []
    for rep in range(CV_REPEATS):
        # Mask a random subset of observed entries again
        n_mask = int(CV_FRAC_MASK * n_total_obs)
        chosen = np.random.choice(n_total_obs, size=n_mask, replace=False)
        rows = obs_indices[0][chosen]
        cols = obs_indices[1][chosen]

        X_cv = X_std_orig.copy()
        X_cv[rows, cols] = np.nan

        pca_cv, scores_cv, X_cv_em = run_em_pca(
            X_cv,
            n_components=K,
            max_iter=EM_MAX_ITER,
            tol=EM_TOL,
            random_state=RANDOM_STATE + 100 * K + rep,
        )

        true_vals = X_std_orig[rows, cols]
        preds = X_cv_em[rows, cols]
        rmse_list.append(reconstruction_rmse(true_vals, preds))

    cv_error_K = float(np.mean(rmse_list))
    cv_errors.append(cv_error_K)
    print(f"    Avg RMSE for K={K}: {cv_error_K:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(Ks, cv_errors, marker="o")
plt.xlabel("Number of components K")
plt.ylabel("CV reconstruction RMSE (std units)")
plt.title("Cross-validated reconstruction error vs. #components (EM-PCA)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# ------------- SCALING SENSITIVITY ANALYSIS ----------------
# ============================================================

print("\nScaling sensitivity: PCA on standardised vs raw data...")

# Standardised (already have X_std with EM-PCA reconstruction)
X_std_full = X_em_filled  # filled version in standardised space
pca_std = PCA(n_components=1, random_state=RANDOM_STATE)
scores_std = pca_std.fit_transform(X_std_full)
loadings_std = pca_std.components_[0, :]

# Raw (unstandardised) data: impute means in raw space
X_raw_imp = mean_impute(X_raw)
pca_raw = PCA(n_components=1, random_state=RANDOM_STATE)
scores_raw = pca_raw.fit_transform(X_raw_imp)
loadings_raw = pca_raw.components_[0, :]

# Plot absolute loadings for PC1
plt.figure(figsize=(10, 6))
x = np.arange(len(feature_names))
width = 0.35

plt.bar(x - width / 2, np.abs(loadings_std), width, label="Standardised PCA")
plt.bar(x + width / 2, np.abs(loadings_raw), width, label="Raw PCA")
plt.xticks(x, feature_names, rotation=45, ha="right")
plt.ylabel("Absolute PC1 loading")
plt.title("Scaling sensitivity: standardised vs raw PCA (PC1 loadings)")
plt.legend()
plt.grid(True, axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

print("\nDone. All figures generated.")
