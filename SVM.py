# SVM_SVR_implementation.py
# SVR for multivariate regression on ENB2012_data.xlsx (Y1 from X1..X8)

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------- 1) Load dataset (Excel in repo root) ----------
DATA_PATH = Path("ENB2012_data.xlsx")
df = pd.read_excel(DATA_PATH, engine="openpyxl")

# Clean headers and standardize names
df.columns = [c.strip() for c in df.columns]
rename_map = {
    "Relative Compactness": "X1",
    "Surface Area": "X2",
    "Wall Area": "X3",
    "Roof Area": "X4",
    "Overall Height": "X5",
    "Orientation": "X6",
    "Glazing Area": "X7",
    "Glazing Area Distribution": "X8",
    "Heating Load": "Y1",
    "Cooling Load": "Y2",
}
df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

# ---------- 2) Features & target ----------
feature_cols = [f"X{i}" for i in range(1, 9)]
target_col = "Y1"  # change to "Y2" for Cooling Load

# Ensure numeric dtypes
df[feature_cols + [target_col]] = df[feature_cols + [target_col]].apply(pd.to_numeric, errors="coerce")

# Drop rows where target is NaN (should not happen in this dataset)
df = df.dropna(subset=[target_col]).reset_index(drop=True)

X = df[feature_cols].copy()
y = df[target_col].values

print(f"NaNs in X before imputation: {int(X.isna().sum().sum())}")

# ---------- 3) Train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ---------- 4) SVR pipeline (Imputer + Scaler + RBF SVR) ----------
svr_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf")),
])

# Utility: RMSE for older sklearn (no 'squared' arg)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# ---------- 5) Fit default SVR ----------
svr_pipe.fit(X_train, y_train)
pred_def = svr_pipe.predict(X_test)

print("\n=== SVR (default) ===")
print("MAE :", mean_absolute_error(y_test, pred_def))
print("RMSE:", rmse(y_test, pred_def))
print("R^2 :", r2_score(y_test, pred_def))

# ---------- 6) Brief tuning (small grid, CV=5) ----------
param_grid = {
    "svr__C": [1, 10, 100],
    "svr__epsilon": [0.1, 0.5, 1.0],
    "svr__gamma": ["scale", 0.1, 0.01],
}
gcv = GridSearchCV(
    svr_pipe, param_grid=param_grid,
    scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1
)
gcv.fit(X_train, y_train)

best_svr = gcv.best_estimator_
pred_best = best_svr.predict(X_test)

print("\n=== SVR (tuned) ===")
print("Best params:", gcv.best_params_)
print("MAE :", mean_absolute_error(y_test, pred_best))
print("RMSE:", rmse(y_test, pred_best))
print("R^2 :", r2_score(y_test, pred_best))

# ---------- 7) Quick diagnostic plot ----------
plt.figure(figsize=(6, 4))
plt.scatter(y_test, pred_best, alpha=0.7)
lims = [min(y_test.min(), pred_best.min()), max(y_test.max(), pred_best.max())]
plt.plot(lims, lims, "r--")
plt.xlabel("Actual Y1 (Heating Load)")
plt.ylabel("Predicted Y1 (SVR tuned)")
plt.title("SVR: Predicted vs Actual")
plt.tight_layout()
plt.show()
