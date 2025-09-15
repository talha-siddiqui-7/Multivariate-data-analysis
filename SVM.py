# SVM_SVR_implementation_both_targets_headless.py
# SVR for multivariate regression on ENB2012_data.xlsx (runs Y1 and Y2)
# - Reads Excel with openpyxl
# - Uses headless Matplotlib backend (no Tkinter)
# - Saves plots to PNG instead of showing GUI windows

import matplotlib
matplotlib.use("Agg")  # headless backend to avoid Tkinter warnings
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------- Load dataset (Excel in repo root) ----------
DATA_PATH = Path("ENB2012_data.xlsx")
df = pd.read_excel(DATA_PATH, engine="openpyxl")

# Clean headers and standardize names (covers UCI column labels)
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

feature_cols = [f"X{i}" for i in range(1, 9)]

# Utility: RMSE (compatible with older sklearn)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Common pipeline (Imputer + Scaler + RBF SVR)
base_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf")),
])

param_grid = {
    "svr__C": [1, 10, 100],
    "svr__epsilon": [0.1, 0.5, 1.0],
    "svr__gamma": ["scale", 0.1, 0.01],
}

def run_target(target_col: str):
    # Ensure numeric, drop rows with missing target (should be none in UCI file)
    cols = feature_cols + [target_col]
    df_local = df.copy()
    df_local[cols] = df_local[cols].apply(pd.to_numeric, errors="coerce")
    data = df_local.dropna(subset=[target_col]).reset_index(drop=True)

    X = data[feature_cols].copy()
    y = data[target_col].values

    print(f"\n=== Running target: {target_col} ===")
    print(f"NaNs in X before imputation: {int(X.isna().sum().sum())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Default SVR
    pipe = base_pipe
    pipe.fit(X_train, y_train)
    pred_def = pipe.predict(X_test)

    print(f"\n=== {target_col}: SVR (default) ===")
    print("MAE :", mean_absolute_error(y_test, pred_def))
    print("RMSE:", rmse(y_test, pred_def))
    print("R^2 :", r2_score(y_test, pred_def))

    # Tuned SVR
    gcv = GridSearchCV(
        base_pipe, param_grid=param_grid,
        scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1
    )
    gcv.fit(X_train, y_train)
    best_model = gcv.best_estimator_
    pred_best = best_model.predict(X_test)

    print(f"\n=== {target_col}: SVR (tuned) ===")
    print("Best params:", gcv.best_params_)
    print("MAE :", mean_absolute_error(y_test, pred_best))
    print("RMSE:", rmse(y_test, pred_best))
    print("R^2 :", r2_score(y_test, pred_best))

    # Save plot (Predicted vs Actual)
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, pred_best, alpha=0.7)
    lims = [min(y_test.min(), pred_best.min()), max(y_test.max(), pred_best.max())]
    plt.plot(lims, lims, "r--")
    label = "Heating Load" if target_col == "Y1" else "Cooling Load"
    plt.xlabel(f"Actual {label}")
    plt.ylabel(f"Predicted {label} (SVR tuned)")
    plt.title(f"SVR: Predicted vs Actual — {label}")
    plt.tight_layout()
    out_name = f"svr_pred_actual_{target_col.lower()}.png"  # e.g., svr_pred_actual_y1.png
    plt.savefig(out_name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot -> {out_name}")

# Run both targets
run_target("Y1")
run_target("Y2")
