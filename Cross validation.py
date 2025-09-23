# choose_splits_multi_k.py
# Compare Hold-out vs LOOCV vs multiple StratifiedKFold(k) settings on Breast Cancer dataset
# Saves: one plot per k (bar chart mean ± std) + a CSV summary

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless (no GUI)
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# Config
# -----------------------------
K_LIST = [3, 5, 7, 10]   # <- choose the k values you want to demo
TEST_SIZE = 0.20
RANDOM_STATE = 42

# -----------------------------
# Helpers
# -----------------------------
def summarize_scores(name, scores, start_time):
    duration = time.time() - start_time
    return {
        "method": name,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "n_folds": int(len(scores)),
        "time_sec": duration,
    }

def barplot_mean_std(methods, means, stds, title, outpath):
    x = np.arange(len(methods))
    plt.figure(figsize=(7.5, 4.2))
    plt.bar(x, means, yerr=stds, capsize=6)
    plt.xticks(x, methods, rotation=0)
    plt.ylabel("Accuracy (mean ± std)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# -----------------------------
# Data + model (pipeline to avoid leakage)
# -----------------------------
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
clf_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, solver="lbfgs")),
])

# -----------------------------
# Baseline: Hold-out (single split)
# -----------------------------
t0 = time.time()
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=TEST_SIZE, shuffle=True, random_state=RANDOM_STATE
)
clf_pipe.fit(X_tr, y_tr)
y_hat = clf_pipe.predict(X_te)
holdout_acc = accuracy_score(y_te, y_hat)
holdout_summary = summarize_scores(
    f"Hold-out ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)})",
    np.array([holdout_acc]),
    t0
)

# -----------------------------
# LOOCV (constant across k)
# -----------------------------
t0 = time.time()
loo = LeaveOneOut()
loo_scores = cross_val_score(clf_pipe, X, y, cv=loo, scoring="accuracy")
loo_summary = summarize_scores("LOOCV", loo_scores, t0)

# -----------------------------
# Loop over StratifiedKFold(k) values
# -----------------------------
rows = []
plots = []
for k in K_LIST:
    # Stratified K-Fold with this k
    t0 = time.time()
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    skf_scores = cross_val_score(clf_pipe, X, y, cv=skf, scoring="accuracy")
    skf_summary = summarize_scores(f"StratifiedKFold (k={k})", skf_scores, t0)

    # Collect rows for master table
    rows.append({
        "variant": f"k={k}",
        "method": holdout_summary["method"],
        "mean": holdout_summary["mean"],
        "std": holdout_summary["std"],
        "n_folds": holdout_summary["n_folds"],
        "time_sec": holdout_summary["time_sec"],
    })
    rows.append({
        "variant": f"k={k}",
        "method": skf_summary["method"],
        "mean": skf_summary["mean"],
        "std": skf_summary["std"],
        "n_folds": skf_summary["n_folds"],
        "time_sec": skf_summary["time_sec"],
    })
    rows.append({
        "variant": f"k={k}",
        "method": loo_summary["method"],
        "mean": loo_summary["mean"],
        "std": loo_summary["std"],
        "n_folds": loo_summary["n_folds"],
        "time_sec": loo_summary["time_sec"],
    })

    # Make a per-k comparison plot (Hold-out vs Stratified k vs LOOCV)
    methods = [holdout_summary["method"], skf_summary["method"], loo_summary["method"]]
    means   = [holdout_summary["mean"],   skf_summary["mean"],   loo_summary["mean"]]
    stds    = [holdout_summary["std"],    skf_summary["std"],    loo_summary["std"]]
    out_png = f"classification_comparison_k{k}.png"
    title   = f"Breast Cancer — Split Methods Comparison (k={k})"
    barplot_mean_std(methods, means, stds, title, out_png)
    plots.append(out_png)
    print(f"Saved plot -> {out_png}")

# -----------------------------
# Save a consolidated summary
# -----------------------------
summary_df = pd.DataFrame(rows)
summary_csv = "classification_split_summary.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"\nSaved summary -> {summary_csv}")

# Pretty print per-k block in console
for k in K_LIST:
    block = summary_df[summary_df["variant"] == f"k={k}"].copy()
    print("\n" + "="*40)
    print(f"Variant k={k}")
    print("="*40)
    for _, r in block.iterrows():
        print(f"{r['method']:<24} | mean={r['mean']:.4f}  std={r['std']:.4f}  folds={int(r['n_folds'])}  time={r['time_sec']:.2f}s")

print("\nArtifacts saved:")
for p in plots:
    print(f"  - {p}")
print(f"  - {summary_csv}")
