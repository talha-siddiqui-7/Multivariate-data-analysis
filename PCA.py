# --- 0) Imports & config ------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer

np.set_printoptions(suppress=True, linewidth=140)
plt.rcParams["figure.figsize"] = (7,4)
RNG = np.random.default_rng(42)



PATH = "Copy of AirQualityUCI.csv"   # your file

# ✅ Read as comma CSV (no semicolon / comma-decimal here)
df = pd.read_csv(PATH)

# ✅ Drop the two empty trailing columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed", na=False)]

# ✅ Convert UCI’s sentinel -200 to NaN
df = df.replace(-200, np.nan)

# (optional) build a datetime index
dt = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")

# Now select variables (these exist in your file)
COLS = ["T","RH","AH","CO(GT)","NOx(GT)","NO2(GT)","C6H6(GT)","PT08.S1(CO)"]
Xdf = df[COLS].astype(float)

print("Shape:", Xdf.shape)
print((Xdf.isna().mean()*100).round(1).sort_values(ascending=False))


# --- 2) Utility: standardize with NaN-aware stats + inverse ------------------
def zscore_nan(X):
    mu = np.nanmean(X, axis=0)
    sig = np.nanstd(X, axis=0, ddof=1)
    Z = (X - mu) / sig
    return Z, mu, sig

def inv_zscore(Z, mu, sig):
    return Z * sig + mu

# Build numpy array; keep a copy for RMSE scoring
X = Xdf.to_numpy()
Z, mu, sig = zscore_nan(X)


# --- 3) Hold-out test mask (hide 10% of the observed entries) ----------------
def make_test_mask(X, frac=0.10, rng=RNG):
    obs = np.where(~np.isnan(X))
    n = int(frac * obs[0].size)
    idx = rng.choice(obs[0].size, size=n, replace=False)
    mask = np.zeros_like(X, dtype=bool)
    mask[obs[0][idx], obs[1][idx]] = True
    X_masked = X.copy()
    X_masked[mask] = np.nan
    return X_masked, mask

Z_masked, test_mask = make_test_mask(Z, frac=0.10, rng=RNG)


# --- 4) Methods: mean→PCA, KNN→PCA, EM-PCA -----------------------------------
def pca_mean_impute(Zin, n_components=3):
    imp = SimpleImputer(strategy="mean")
    Zimp = imp.fit_transform(Zin)
    Zc = Zimp - Zimp.mean(axis=0, keepdims=True)
    pca = PCA(n_components=n_components, svd_solver='full', random_state=0)
    scores = pca.fit_transform(Zc)
    loadings = pca.components_.T
    Zrec = scores @ loadings.T + Zimp.mean(axis=0, keepdims=True)
    evr = pca.explained_variance_ratio_
    return Zrec, evr, loadings, scores

def pca_knn_impute(Zin, n_components=3, n_neighbors=5):
    imp = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    Zimp = imp.fit_transform(Zin)
    Zc = Zimp - Zimp.mean(axis=0, keepdims=True)
    pca = PCA(n_components=n_components, svd_solver='full', random_state=0)
    scores = pca.fit_transform(Zc)
    loadings = pca.components_.T
    Zrec = scores @ loadings.T + Zimp.mean(axis=0, keepdims=True)
    evr = pca.explained_variance_ratio_
    return Zrec, evr, loadings, scores

def empca(Zin, n_components=3, max_iter=200, tol=1e-6, rng=RNG):
    """
    EM-PCA: iteratively reconstruct missing entries using a low-rank PCA model.
    Works directly on arrays with NaNs.
    Returns completed Z, reconstruction, EVR, loadings, scores.
    """
    Z = np.array(Zin, dtype=float, copy=True)
    nanmask = np.isnan(Z)

    # Init: column means (for standardized data, close to 0)
    col_means = np.nanmean(Z, axis=0)
    Z[nanmask] = np.take(col_means, np.where(nanmask)[1])

    prev_missing = Z[nanmask].copy()
    for _ in range(max_iter):
        mu = Z.mean(axis=0, keepdims=True)
        Zc = Z - mu
        U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
        V = Vt.T[:, :n_components]     # loadings
        T = U[:, :n_components] * S[:n_components]  # scores
        Zhat = T @ V.T + mu            # reconstruction

        # E-step: update only missing entries
        Z[nanmask] = Zhat[nanmask]

        # Convergence: change on missing values
        delta = np.nanmean((Z[nanmask] - prev_missing)**2)**0.5
        if delta < tol:
            break
        prev_missing = Z[nanmask].copy()

    # Final PCA on completed Z
    mu = Z.mean(axis=0, keepdims=True)
    Zc = Z - mu
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    V = Vt.T[:, :n_components]
    T = U[:, :n_components] * S[:n_components]
    Zhat = T @ V.T + mu
    evr = (S[:n_components]**2) / (S**2).sum()
    return Z, Zhat, evr, V, T


# --- 5) Run all three and score on hidden entries (RMSE) ---------------------
def rmse_on_mask(Yhat, Ytrue, mask):
    return np.sqrt(np.nanmean((Yhat[mask] - Ytrue[mask])**2))

NCOMP = 3  # use 3 PCs for compactness (change if you like)

# A) Mean → PCA
Zrec_A, evr_A, L_A, T_A = pca_mean_impute(Z_masked, n_components=NCOMP)
Xrec_A = inv_zscore(Zrec_A, mu, sig)
rmse_A = rmse_on_mask(Xrec_A, X, test_mask)

# B) KNN → PCA
Zrec_B, evr_B, L_B, T_B = pca_knn_impute(Z_masked, n_components=NCOMP, n_neighbors=5)
Xrec_B = inv_zscore(Zrec_B, mu, sig)
rmse_B = rmse_on_mask(Xrec_B, X, test_mask)

# C) EM-PCA (handles NaNs directly)
Zcomp_C, Zrec_C, evr_C, L_C, T_C = empca(Z_masked, n_components=NCOMP)
Xrec_C = inv_zscore(Zrec_C, mu, sig)
rmse_C = rmse_on_mask(Xrec_C, X, test_mask)

print(f"RMSE on hidden entries (original units):\n  Mean→PCA: {rmse_A:.3f}\n  KNN→PCA : {rmse_B:.3f}\n  EM-PCA  : {rmse_C:.3f}")


# --- 6) Plots: missingness map, RMSE bars, scree, loadings, biplot, PC1 line -
# (1) Missingness heatmap
plt.figure()
plt.imshow(np.isnan(X), aspect='auto', interpolation='nearest')
plt.xlabel("Variables"); plt.ylabel("Rows"); plt.title("Missingness map (NaN=1)")
plt.xticks(range(len(COLS)), COLS, rotation=45)
plt.tight_layout(); plt.show()

# (2) RMSE bar chart
plt.figure()
methods = ["Mean→PCA", "KNN→PCA", "EM-PCA"]
rmses = [rmse_A, rmse_B, rmse_C]
plt.bar(methods, rmses)
plt.ylabel("RMSE on hidden entries"); plt.title("Imputation accuracy comparison")
plt.tight_layout(); plt.show()

# (3) Scree (explained variance ratio)
plt.figure()
for evr, lab in [(evr_A,"Mean→PCA"), (evr_B,"KNN→PCA"), (evr_C,"EM-PCA")]:
    plt.plot(np.arange(1, len(evr)+1), np.cumsum(evr), marker='o', label=lab)
plt.xlabel("Number of PCs"); plt.ylabel("Cumulative explained variance")
plt.legend(); plt.title("Scree (cumulative EVR)")
plt.tight_layout(); plt.show()

# (4) Loadings heatmap (EM-PCA model)
plt.figure()
plt.imshow(L_C[:, :NCOMP], aspect='auto', interpolation='nearest')
plt.colorbar(); plt.yticks(range(len(COLS)), COLS)
plt.xticks(range(NCOMP), [f"PC{i}" for i in range(1, NCOMP+1)])
plt.title("Loadings (EM-PCA)"); plt.tight_layout(); plt.show()

# (5) PC1–PC2 biplot (scores) colored by month (if Date+Time available)
if {'Date','Time'}.issubset(df.columns):
    # Build a datetime index
    dt = pd.to_datetime(df['Date'] + " " + df['Time'], errors='coerce', dayfirst=True)
    months = dt.dt.month.to_numpy()
    valid = ~np.isnan(T_C).any(axis=1)
    plt.figure()
    sc = plt.scatter(T_C[valid,0], T_C[valid,1], c=months[valid], s=6)
    plt.xlabel("PC1 score"); plt.ylabel("PC2 score"); plt.title("Scores (EM-PCA)")
    cbar = plt.colorbar(sc); cbar.set_label("Month")
    plt.tight_layout(); plt.show()

# (6) PC1 time series (EM-PCA)
plt.figure()
plt.plot(T_C[:,0])
plt.title("PC1 score over time (EM-PCA)"); plt.xlabel("Row index"); plt.ylabel("PC1 score")
plt.tight_layout(); plt.show()


