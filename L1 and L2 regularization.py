# ================================================================
# L1 vs L2: Demonstrating WHEN each is appropriate
# Datasets:
#   A) Synthetic sparse regression (make_regression)  -> L1 (Lasso) shines
#   B) Diabetes (real, diffuse signal, moderate corr) -> L2 (Ridge) is safer/stable
# What you’ll get:
#   - RMSE vs #non-zero coefficients (parsimony vs accuracy)
#   - Selection precision/recall/F1 (synthetic only)
#   - Stability (Jaccard of selected sets across resamples, Diabetes)
#   - Regularization paths (L1 zeros vs L2 shrinkage)
#   - Correlation heatmap (Diabetes)
# CV: Stratified K-Fold for regression via y-quantile bins
# ================================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression, load_diabetes
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso, RidgeCV, Ridge
from sklearn.linear_model import lasso_path
from sklearn.metrics import mean_squared_error, r2_score

RANDOM_STATE = 42
rng = np.random.RandomState(RANDOM_STATE)

# -----------------------------
# Utilities
# -----------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def stratified_kfold_regression_splits(y, n_splits=10, n_bins=5, shuffle=True, random_state=RANDOM_STATE):
    """
    Make 'stratified' folds for regression by binning y into quantiles.
    Returns a list of (train_idx, val_idx) pairs usable as cv= in sklearn.
    """
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(y, qs))
    # Make sure we have at least 3 bins (avoid degenerate splits if many ties)
    while len(edges) - 1 < 3 and n_bins > 3:
        n_bins -= 1
        edges = np.unique(np.quantile(y, np.linspace(0, 1, n_bins + 1)))
    y_bins = np.digitize(y, edges[1:-1], right=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return list(skf.split(np.zeros_like(y_bins), y_bins))

def lasso_min_and_1se(Xtr, ytr, cv_splits, n_alphas=1000, max_iter=20000):
    """
    Fit LassoCV and extract:
      - alpha_min (min mean CV MSE)
      - alpha_1se (largest alpha whose mean CV MSE <= min + 1*SE)
    """
    lcv = LassoCV(cv=cv_splits, random_state=RANDOM_STATE, n_alphas=n_alphas, max_iter=max_iter)
    lcv.fit(Xtr, ytr)

    mean_mse = lcv.mse_path_.mean(axis=1)
    se_mse   = lcv.mse_path_.std(axis=1) / np.sqrt(lcv.mse_path_.shape[1])
    i_min = np.argmin(mean_mse)
    thresh = mean_mse[i_min] + se_mse[i_min]
    valid = np.where(mean_mse <= thresh)[0]
    alpha_min = lcv.alpha_
    alpha_1se = lcv.alphas_[valid[np.argmax(lcv.alphas_[valid])]]
    return alpha_min, alpha_1se, lcv

def selection_metrics(true_support, coef, tol=1e-8):
    sel = np.abs(coef) > tol
    tp = np.sum(sel & true_support)
    fp = np.sum(sel & ~true_support)
    fn = np.sum((~sel) & true_support)
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec  = tp/(tp+fn) if (tp+fn) else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    return prec, rec, f1, int(sel.sum())

def jaccard(A, B):
    U = A | B
    return len(A & B) / len(U) if U else 1.0

# -----------------------------
# Plot helpers (one figure per plot)
# -----------------------------
def plot_rmse_vs_nnz(nnz_l1, rmse_l1, rmse_ridge, p_features, title):
    plt.figure()
    plt.plot(nnz_l1, rmse_l1, marker='o', label='L1/Lasso (vary α)')
    plt.scatter([p_features], [rmse_ridge], label='L2/Ridge (keeps all)', marker='x')
    plt.xlabel('# Non-zero coefficients')
    plt.ylabel('Test RMSE')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

def plot_l1_path(X, y, title, n_alphas=100):
    Xs = StandardScaler().fit_transform(X)
    alphas, coefs, _ = lasso_path(Xs, y, n_alphas=n_alphas, random_state=RANDOM_STATE)
    plt.figure()
    for i in range(coefs.shape[0]):
        plt.plot(np.log(alphas), coefs[i, :])
    plt.xlabel('log(alpha)')
    plt.ylabel('Coefficient value')
    plt.title(title + ' — L1 coefficient paths')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

def plot_l2_path(X, y, alphas, title):
    Xs = StandardScaler().fit_transform(X)
    coefs = []
    for a in alphas:
        coefs.append(Ridge(alpha=a).fit(Xs, y).coef_)
    coefs = np.array(coefs).T  # shape: p x len(alphas)
    plt.figure()
    for i in range(coefs.shape[0]):
        plt.plot(np.log(alphas), coefs[i, :])
    plt.xlabel('log(alpha)')
    plt.ylabel('Coefficient value')
    plt.title(title + ' — L2 coefficient “paths”')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

def plot_corr_heatmap(X, feature_names, title):
    C = np.corrcoef(X, rowvar=False)
    plt.figure()
    plt.imshow(C, interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
    plt.yticks(range(len(feature_names)), feature_names)
    plt.title(title + ' — Feature correlation')
    plt.tight_layout()

# -----------------------------
# A) Synthetic sparse dataset
# -----------------------------
def run_synthetic_sparse_demo():
    print("\n=== A) Synthetic sparse (expect L1 to shine) ===")
    n_samples, n_features, n_informative, noise = 600, 200, 12, 8.0

    X, y, coef_true = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        coef=True,
        random_state=RANDOM_STATE,
        effective_rank=None  # low collinearity
    )
    true_support = np.abs(coef_true) > 1e-12

    # Train/test split + scaling
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    scaler = StandardScaler().fit(X_tr)
    Xtr = scaler.transform(X_tr)
    Xte = scaler.transform(X_te)

    # Stratified CV on y (binned)
    cv_splits = stratified_kfold_regression_splits(y_tr, n_splits=10, n_bins=5)

    # L1: CV to pick alpha_min and alpha_1se
    a_min, a_1se, lcv = lasso_min_and_1se(Xtr, y_tr, cv_splits)
    yhat_l1 = lcv.predict(Xte)
    rmse_l1 = rmse(y_te, yhat_l1)
    r2_l1 = r2_score(y_te, yhat_l1)

    # L2: CV to pick alpha
    ridge = RidgeCV(alphas=np.logspace(-4, 4, 200), cv=cv_splits, scoring='neg_mean_squared_error')
    ridge.fit(Xtr, y_tr)
    yhat_l2 = ridge.predict(Xte)
    rmse_l2 = rmse(y_te, yhat_l2)
    r2_l2 = r2_score(y_te, yhat_l2)

    # Selection quality (synthetic only)
    prec, rec, f1, k = selection_metrics(true_support, lcv.coef_)

    print(f"L1/Lasso (α_min={a_min:.4g}) -> RMSE={rmse_l1:.4f} | R²={r2_l1:.4f} | "
          f"Selected {k}/{n_features} | Precision={prec:.3f} | Recall={rec:.3f} | F1={f1:.3f}")
    print(f"L2/Ridge (α={ridge.alpha_:.4g}) -> RMSE={rmse_l2:.4f} | R²={r2_l2:.4f} | "
          f"Selected ~{n_features} (Ridge doesn’t prune)")

    # RMSE vs #non-zeros curve for L1, plus Ridge point
    lasso_alphas = np.logspace(-3, 1, 24)  # sweep
    nnz_list, rmse_list = [], []
    for a in lasso_alphas:
        m = Lasso(alpha=a, max_iter=20000, random_state=RANDOM_STATE).fit(Xtr, y_tr)
        rm = rmse(y_te, m.predict(Xte))
        nnz = int(np.sum(np.abs(m.coef_) > 1e-8))
        rmse_list.append(rm); nnz_list.append(nnz)

    plot_rmse_vs_nnz(nnz_list, rmse_list, rmse_l2, n_features,
                     title="Synthetic sparse — RMSE vs sparsity")

    # Paths
    plot_l1_path(X_tr, y_tr, "Synthetic sparse")
    ridge_alphas_for_path = np.logspace(-4, 2, 24)
    plot_l2_path(X_tr, y_tr, ridge_alphas_for_path, "Synthetic sparse")

# -----------------------------
# B) Diabetes dataset (real)
# -----------------------------
def run_diabetes_demo():
    print("\n=== B) Diabetes (real; diffuse signal, moderate correlation) ===")
    data = load_diabetes()
    X, y = data.data, data.target
    feature_names = list(data.feature_names) if hasattr(data, 'feature_names') else [f"x{i}" for i in range(X.shape[1])]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    scaler = StandardScaler().fit(X_tr)
    Xtr = scaler.transform(X_tr)
    Xte = scaler.transform(X_te)

    cv_splits = stratified_kfold_regression_splits(y_tr, n_splits=10, n_bins=5)

    # L1: choose alpha_min and alpha_1se
    a_min, a_1se, lcv = lasso_min_and_1se(Xtr, y_tr, cv_splits)
    yhat_l1_min = lcv.predict(Xte)
    rmse_l1_min = rmse(y_te, yhat_l1_min)
    r2_l1_min = r2_score(y_te, yhat_l1_min)
    nnz_l1_min = int(np.sum(np.abs(lcv.coef_) > 1e-8))

    l1se_model = Lasso(alpha=a_1se, max_iter=20000, random_state=RANDOM_STATE).fit(Xtr, y_tr)
    yhat_l1_1se = l1se_model.predict(Xte)
    rmse_l1_1se = rmse(y_te, yhat_l1_1se)
    r2_l1_1se = r2_score(y_te, yhat_l1_1se)
    nnz_l1_1se = int(np.sum(np.abs(l1se_model.coef_) > 1e-8))

    # L2: RidgeCV
    ridge = RidgeCV(alphas=np.logspace(-4, 4, 200), cv=cv_splits, scoring='neg_mean_squared_error')
    ridge.fit(Xtr, y_tr)
    yhat_r = ridge.predict(Xte)
    rmse_r = rmse(y_te, yhat_r)
    r2_r = r2_score(y_te, yhat_r)

    print(f"L2/Ridge (α={ridge.alpha_:.4g}) -> RMSE={rmse_r:.4f} | R²={r2_r:.4f} | keeps all features")
    print(f"L1/Lasso α_min ({a_min:.4g}) -> RMSE={rmse_l1_min:.4f} | R²={r2_l1_min:.4f} | non-zeros={nnz_l1_min}/{X.shape[1]}")
    print(f"L1/Lasso α_1se ({a_1se:.4g}) -> RMSE={rmse_l1_1se:.4f} | R²={r2_l1_1se:.4f} | non-zeros={nnz_l1_1se}/{X.shape[1]}")

    # RMSE vs #non-zeros (sweep α for L1), Ridge point
    lasso_alphas = np.logspace(-3, 1, 24)
    nnz_list, rmse_list = [], []
    for a in lasso_alphas:
        m = Lasso(alpha=a, max_iter=20000, random_state=RANDOM_STATE).fit(Xtr, y_tr)
        rm = rmse(y_te, m.predict(Xte))
        nnz = int(np.sum(np.abs(m.coef_) > 1e-8))
        rmse_list.append(rm); nnz_list.append(nnz)

    plot_rmse_vs_nnz(nnz_list, rmse_list, rmse_r, X.shape[1],
                     title="Diabetes — RMSE vs sparsity")

    # Paths + correlation heatmap
    plot_l1_path(X_tr, y_tr, "Diabetes")
    ridge_alphas_for_path = np.logspace(-4, 2, 24)
    plot_l2_path(X_tr, y_tr, ridge_alphas_for_path, "Diabetes")
    plot_corr_heatmap(X_tr, feature_names, "Diabetes")

    # Stability: Jaccard similarity of L1-selected sets across resamples (using α_min per split)
    print("\nStability (Diabetes, L1 selection sets across 20 random splits):")
    sets = []
    for rep in range(20):
        rs = rng.randint(0, 10_000)
        Xtr_, Xte_, ytr_, yte_ = train_test_split(X, y, test_size=0.3, random_state=rs)
        sc_ = StandardScaler().fit(Xtr_)
        Xtr_s = sc_.transform(Xtr_)
        cv_ = stratified_kfold_regression_splits(ytr_, n_splits=10, n_bins=5, random_state=rs)
        a_min_, a_1se_, lcv_ = lasso_min_and_1se(Xtr_s, ytr_, cv_)
        sel_idx = set(np.where(np.abs(lcv_.coef_) > 1e-8)[0])
        sets.append(sel_idx)

    # Average pairwise Jaccard
    from itertools import combinations
    J = [jaccard(a, b) for a, b in combinations(sets, 2)]
    print(f"Mean Jaccard = {np.mean(J):.3f} ± {np.std(J, ddof=1):.3f} "
          f"(higher = more stable; often modest here)")

# -----------------------------
# Run everything
# -----------------------------
if __name__ == "__main__":
    run_synthetic_sparse_demo()
    run_diabetes_demo()
    plt.show()
