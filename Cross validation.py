# choose_splits_demo.py
# Demonstrates: Hold-out vs Stratified K-Fold vs LOOCV (classification)
#
# Run: python choose_splits_demo.py

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save plots to files (no GUI popups)
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# Helpers
# -----------------------------
def print_header(title):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

def summarize_scores(name, scores, start_time):
    duration = time.time() - start_time
    print(f"{name} -> mean: {np.mean(scores):.4f}, std: {np.std(scores):.4f}, n={len(scores)}, time={duration:.2f}s")
    return {"method": name, "mean": float(np.mean(scores)), "std": float(np.std(scores)), "n": len(scores), "time_sec": duration}

# -----------------------------
# Classification: Hold-out, Stratified K-Fold, LOOCV
# Dataset: Breast Cancer Wisconsin (binary classification)
# -----------------------------
print_header("Classification splits (Breast Cancer)")

# Load data
X_bc, y_bc = load_breast_cancer(return_X_y=True, as_frame=True)

# Model (fast, stable): Logistic Regression with scaling in a Pipeline
clf_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, solver="lbfgs"))
])

# 1) Hold-out (single split)
start = time.time()
X_tr, X_te, y_tr, y_te = train_test_split(X_bc, y_bc, test_size=0.2, shuffle=True, random_state=42)  # standard hold-out
clf_pipe.fit(X_tr, y_tr)
y_pred = clf_pipe.predict(X_te)
holdout_acc = accuracy_score(y_te, y_pred)
holdout_summary = summarize_scores("Hold-out (80/20)", np.array([holdout_acc]), start)

# 2) Stratified K-Fold CV (k=5)
start = time.time()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf_scores = cross_val_score(clf_pipe, X_bc, y_bc, cv=skf, scoring="accuracy")
skf_summary = summarize_scores("StratifiedKFold (k=5)", skf_scores, start)

# 3) LOOCV (Leave-One-Out CV)
# Note: 569 fits — okay for this small dataset
start = time.time()
loo = LeaveOneOut()
loo_scores = cross_val_score(clf_pipe, X_bc, y_bc, cv=loo, scoring="accuracy")
loo_summary = summarize_scores("LOOCV", loo_scores, start)

# Plot summary for classification
cls_summary_df = pd.DataFrame([holdout_summary, skf_summary, loo_summary])
plt.figure(figsize=(7, 4))
plt.errorbar(cls_summary_df["method"], cls_summary_df["mean"], yerr=cls_summary_df["std"], fmt="o", capsize=5)
plt.ylabel("Accuracy (mean ± std)")
plt.title("Classification: Split Strategy Comparison")
plt.tight_layout()
plt.savefig("classification_split_comparison.png", dpi=150)
plt.close()
print("Saved plot -> classification_split_comparison.png")

# -----------------------------
# Console summary
# -----------------------------
print_header("SUMMARY")
for row in [holdout_summary, skf_summary, loo_summary]:
    print(f"{row['method']:>22}: mean={row['mean']:.4f}, std={row['std']:.4f}, time={row['time_sec']:.2f}s")

print("\nArtifacts saved:")
print("  - classification_split_comparison.png")
