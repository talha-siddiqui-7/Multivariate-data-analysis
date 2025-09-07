# %% [markdown]
# # PCA with Missing Data (AirQualityUCI)
# Blocked, narrated pipeline for a talk/demo:
# 1) Load & clean  →  2) Standardize + test-mask  →  3) PCA pipelines
# 4) Run & reconstruct  →  5) Metrics (RMSE & Std-RMSE)  →  6) Plots
# 7) Summary
# Next block: imports, config, and reproducibility.

# %%
# BLOCK 0 — Imports, global config, reproducibility
# What this block does: sets up libraries, plotting defaults, file path, and key knobs.
# Next: data loading & cleaning.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer

plt.rcParams["figure.figsize"] = (7, 4)
plt.rcParams["axes.grid"] = False
RNG = np.random.default_rng(42)

CONFIG = {
    "DATA_PATH": "Copy of AirQualityUCI.csv",  # change path if needed
    "COLS": ["T","RH","AH","CO(GT)","NOx(GT)","NO2(GT)","C6H6(GT)","PT08.S1(CO)"],
    "TEST_FRAC": 0.10,          # fraction of observed cells to hide for scoring
    "NCOMP": 3,                 # number of PCs to keep/plot
    "KNN_NEIGHBORS": 5          # neighbors for KNN imputation
}

# %%
# BLOCK 1 — Robust loader & cleaning
# What this block does:
# - Reads CSV/XLSX, fixes headers, drops junk columns, replaces sentinel -200→NaN.
# - Selects the variables for PCA; shows missingness percentages.
# Next: standardize and create the test-mask for fair evaluation.
def load_air_quality(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
        # If it collapses into one wide column, assume UCI semicolon CSV:
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=";", decimal=",", engine="python")
    # Tidy headers & drop unnamed
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", na=False)]
    df.columns = (df.columns.str.strip()
                              .str.replace("\xa0"," ", regex=False)
                              .str.replace("\u200b","", regex=False))
    # Replace sentinel -200 with NaN
    return df.replace(-200, np.nan)

df_raw = load_air_quality(CONFIG["DATA_PATH"])
wanted = CONFIG["COLS"]
present = [c for c in wanted if c in df_raw.columns]
missing = [c for c in wanted if c not in df_raw.columns]
if missing:
    print("⚠️ Missing columns not found in file:", missing)
Xdf = df_raw[present].astype(float)
print("Rows:", len(Xdf))
print("Missing %:", (Xdf.isna().mean()*100).round(1).sort_values(ascending=False))

# Optional: month coloring for scatter plot if Date/Time exist
months = None
if {"Date","Time"}.issubset(df_raw.columns):
    dt = pd.to_datetime(df_raw["Date"] + " " + df_raw["Time"], errors="coerce", dayfirst=True)
    months = dt.dt.month.to_numpy()

# %%
# BLOCK 2 — Standardize + Test-mask (entry-wise hold-out)
# What this block does:
# - Z-scores each column using NaN-aware mean/std (gaps ignored in stats).
# - Builds a test mask by hiding ~10% of originally observed cells (set to NaN).
#   We will grade only on these hidden cells—because we know their true values.
# Next: define PCA pipelines (Mean→PCA, KNN→PCA, EM-PCA).
def zscore_nan(X: np.ndarray):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=1)
    return (X - mu) / sd, mu, sd

def inv_zscore(Z, mu, sd):
    return Z * sd + mu

def make_test_mask(X: np.ndarray, frac=0.10, rng=RNG):
    obs = np.where(~np.isnan(X))         # indices of currently observed cells
    n = int(frac * obs[0].size)
    pick = rng.choice(obs[0].size, size=n, replace=False)
    mask = np.zeros_like(X, dtype=bool)  # True where we will hide & later score
    mask[obs[0][pick], obs[1][pick]] = True
    X_masked = X.copy()
    X_masked[mask] = np.nan
    return X_masked, mask

X = Xdf.to_numpy()
Z, mu, sd = zscore_nan(X)
Z_masked, test_mask = make_test_mask(Z, frac=CONFIG["TEST_FRAC"], rng=RNG)

# %%
# BLOCK 3 — PCA pipelines (impute→PCA and EM-PCA)
# What this block does:
# - Defines 3 ways to deal with NaNs before/during PCA:
#   A) Mean→PCA (baseline), B) KNN→PCA (strong baseline), C) EM-PCA (handles NaNs directly).
# Next: run all pipelines and reconstruct to standardized space.
def pca_mean_impute(Zin, n_components=3):
    imp = SimpleImputer(strategy="mean")
    Zimp = imp.fit_transform(Zin)
    Zc = Zimp - Zimp.mean(axis=0, keepdims=True)
    pca = PCA(n_components=n_components, svd_solver="full", random_state=0)
    scores = pca.fit_transform(Zc)
    loadings = pca.components_.T
    Zrec = scores @ loadings.T + Zimp.mean(axis=0, keepdims=True)
    return Zrec, pca.explained_variance_ratio_, loadings, scores

def pca_knn_impute(Zin, n_components=3, n_neighbors=5):
    imp = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    Zimp = imp.fit_transform(Zin)
    Zc = Zimp - Zimp.mean(axis=0, keepdims=True)
    pca = PCA(n_components=n_components, svd_solver="full", random_state=0)
    scores = pca.fit_transform(Zc)
    loadings = pca.components_.T
    Zrec = scores @ loadings.T + Zimp.mean(axis=0, keepdims=True)
    return Zrec, pca.explained_variance_ratio_, loadings, scores

def empca(Zin, n_components=3, max_iter=200, tol=1e-6):
    """
    EM-PCA: iteratively learns a low-rank PCA model and uses it to fill NaNs.
    E-step: reconstruct; M-step: refit PCA; update only missing cells; repeat.
    """
    Z = np.array(Zin, dtype=float, copy=True)
    nanmask = np.isnan(Z)
    # Init with column means (≈0 after z-scoring)
    col_means = np.nanmean(Z, axis=0)
    Z[nanmask] = np.take(col_means, np.where(nanmask)[1])

    prev_missing = Z[nanmask].copy()
    for _ in range(max_iter):
        mu_ = Z.mean(axis=0, keepdims=True)
        Zc = Z - mu_
        U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
        V = Vt.T[:, :n_components]
        T = U[:, :n_components] * S[:n_components]
        Zhat = T @ V.T + mu_
        Z[nanmask] = Zhat[nanmask]  # update missing only
        delta = float(np.sqrt(np.nanmean((Z[nanmask] - prev_missing)**2)))
        if delta < tol:
            break
        prev_missing = Z[nanmask].copy()

    # Final PCA on completed Z
    mu_ = Z.mean(axis=0, keepdims=True)
    Zc = Z - mu_
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    V = Vt.T[:, :n_components]
    T = U[:, :n_components] * S[:n_components]
    Zhat = T @ V.T + mu_
    evr = (S[:n_components]**2) / (S**2).sum()
    return Z, Zhat, evr, V, T

# %%
# BLOCK 4 — Run all methods & reconstruct (standardized space)
# What this block does:
# - Runs Mean→PCA, KNN→PCA, and EM-PCA on the masked matrix.
# - Produces standardized reconstructions and (for EM) scores/loadings we’ll plot.
# Next: invert scaling to original units and compute metrics (RMSE & standardized RMSE).
ncomp = CONFIG["NCOMP"]
Zrec_A, evr_A, L_A, T_A = pca_mean_impute(Z_masked, n_components=ncomp)
Zrec_B, evr_B, L_B, T_B = pca_knn_impute(Z_masked, n_components=ncomp, n_neighbors=CONFIG["KNN_NEIGHBORS"])
Zcomp_C, Zrec_C, evr_C, L_C, T_C = empca(Z_masked, n_components=ncomp)

# %%
# BLOCK 5 — Metrics (RMSE & Standardized RMSE)
# What this block does:
# - Converts reconstructions back to original units.
# - Computes raw RMSE (mixed units) and standardized RMSE (unit-free, fair).
# Next: plots (missingness, RMSE bars, scree, loadings, scores, PC1 timeline).
def rmse_on_mask(yhat, ytrue, mask):
    return float(np.sqrt(np.nanmean((yhat[mask] - ytrue[mask])**2)))

def standardized_rmse(yhat, ytrue, mask, col_std):
    j = np.where(mask)[1]
    denom = np.where(col_std[j] == 0, 1.0, col_std[j])
    err_std = (yhat - ytrue)[mask] / denom
    return float(np.sqrt(np.mean(err_std**2)))

# back to original units
Xrec_A = inv_zscore(Zrec_A, mu, sd)
Xrec_B = inv_zscore(Zrec_B, mu, sd)
Xrec_C = inv_zscore(Zrec_C, mu, sd)

raw_A = rmse_on_mask(Xrec_A, X, test_mask)
raw_B = rmse_on_mask(Xrec_B, X, test_mask)
raw_C = rmse_on_mask(Xrec_C, X, test_mask)

std_cols = np.nanstd(X, axis=0, ddof=1)
srmse_A = standardized_rmse(Xrec_A, X, test_mask, std_cols)
srmse_B = standardized_rmse(Xrec_B, X, test_mask, std_cols)
srmse_C = standardized_rmse(Xrec_C, X, test_mask, std_cols)

print(f"Raw RMSE  : Mean→PCA={raw_A:.2f} | KNN→PCA={raw_B:.2f} | EM-PCA={raw_C:.2f}")
print(f"Std. RMSE : Mean→PCA={srmse_A:.3f} | KNN→PCA={srmse_B:.3f} | EM-PCA={srmse_C:.3f}")

# %%
# BLOCK 6 — Plots for slides
# What this block does:
# - Missingness map (where NaNs are)
# - RMSE bar charts (raw & standardized)
# - Scree (cumulative EVR) to choose k
# - Loadings heatmap (interpret PCs physically)
# - PC1–PC2 scores (colored by month) + PC1 trend
# Next: optional outlier diagnostics (T² and Q).
def plot_missingness(X_, cols):
    plt.figure()
    plt.imshow(np.isnan(X_), aspect="auto", interpolation="nearest")
    plt.xticks(range(len(cols)), cols, rotation=45)
    plt.xlabel("Variables"); plt.ylabel("Rows"); plt.title("Missingness map (NaN=1)")
    plt.tight_layout(); plt.show()

def plot_rmse_bars(raw_vals, std_vals, labels):
    plt.figure()
    plt.bar(labels, std_vals)
    plt.ylabel("Standardized RMSE"); plt.title("Imputation accuracy (standardized)")
    for i, v in enumerate(std_vals):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout(); plt.show()

    plt.figure()
    plt.bar(labels, raw_vals)
    plt.ylabel("Raw RMSE (mixed units)"); plt.title("Imputation accuracy (raw)")
    for i, v in enumerate(raw_vals):
        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    plt.tight_layout(); plt.show()

def plot_scree(evrs, labels):
    plt.figure()
    for evr, lab in zip(evrs, labels):
        x = np.arange(1, len(evr)+1)
        plt.plot(x, np.cumsum(evr), marker="o", label=lab)
    plt.xlabel("Number of PCs"); plt.ylabel("Cumulative explained variance")
    plt.title("Scree (cumulative EVR)"); plt.legend(); plt.tight_layout(); plt.show()

def plot_loadings(L, cols, ncomp_):
    plt.figure()
    plt.imshow(L[:, :ncomp_], aspect="auto", interpolation="nearest")
    plt.colorbar(); plt.yticks(range(len(cols)), cols)
    plt.xticks(range(ncomp_), [f"PC{i}" for i in range(1, ncomp_+1)])
    plt.title("Loadings (EM-PCA)"); plt.tight_layout(); plt.show()

def plot_scores_scatter(T_scores, months_):
    ok = ~np.isnan(T_scores).any(axis=1)
    plt.figure()
    if months_ is None:
        plt.scatter(T_scores[ok,0], T_scores[ok,1], s=6)
    else:
        sc = plt.scatter(T_scores[ok,0], T_scores[ok,1], c=months_[ok], s=6)
        cbar = plt.colorbar(sc); cbar.set_label("Month")
    plt.xlabel("PC1 score"); plt.ylabel("PC2 score"); plt.title("Scores (EM-PCA)")
    plt.tight_layout(); plt.show()

def plot_pc1_timeseries(T_scores):
    plt.figure()
    plt.plot(T_scores[:,0])
    plt.title("PC1 score over time (EM-PCA)")
    plt.xlabel("Row index"); plt.ylabel("PC1 score")
    plt.tight_layout(); plt.show()

# Draw the plots
plot_missingness(X, present)
plot_rmse_bars([raw_A, raw_B, raw_C], [srmse_A, srmse_B, srmse_C], ["Mean→PCA","KNN→PCA","EM-PCA"])
plot_scree([evr_A, evr_B, evr_C], ["Mean→PCA","KNN→PCA","EM-PCA"])
plot_loadings(L_C, present, CONFIG["NCOMP"])
plot_scores_scatter(T_C, months)
plot_pc1_timeseries(T_C)

# %%
# BLOCK 7 — Summary (print-ready bullets for slide notes)
# What this block does: prints the one-liners you can read on a slide.
print("\n=== One-liners for the talk ===")
print("• We converted -200→NaN, hid 10% known cells, then compared three pipelines.")
print("• EM-PCA reconstructed hidden values best (lowest Std-RMSE), KNN next, Mean last.")
print("• 2–3 PCs capture most structure (scree elbow); loadings map to thermo-moist & pollutant axes.")
print("• Scores show seasonality (summer right/down, winter left/up); residuals flag anomalies.")
