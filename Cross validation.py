# cv_comparison_penguins_fixed2.py
# Compare KFold, StratifiedKFold, GroupKFold (real groups), LOOCV, LeavePOut
# Dataset: Palmer Penguins via OpenML (groups = island)
# Saves: cv_methods_comparison_penguins.png and .csv

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap
import warnings

from sklearn.datasets import fetch_openml
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold, LeaveOneOut, LeavePOut, cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------
# Load dataset (Palmer Penguins)
# -----------------------------
peng = fetch_openml("penguins", version=1, as_frame=True)  # needs internet on 1st run
df = peng.frame

# Normalize column names (OpenML variants)
df = df.rename(columns={
    "culmen_length_mm": "bill_length_mm",
    "culmen_depth_mm": "bill_depth_mm",
})

needed = {
    "species", "island", "bill_length_mm", "bill_depth_mm",
    "flipper_length_mm", "body_mass_g", "sex",
}
missing = needed - set(df.columns)
if missing:
    raise KeyError(f"Dataset columns missing: {missing}. Columns found: {list(df.columns)}")

# Drop missing rows to keep demo simple
df = df[list(needed)].dropna().reset_index(drop=True)

# Target, groups, features
y = df["species"]
groups = df["island"]
X = df.drop(columns=["species", "island"])

numeric_features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
categorical_features = ["sex"]

# -----------------------------
# Build pipeline (preprocess + model)
# -----------------------------
preproc = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

clf_pipe = Pipeline([
    ("preproc", preproc),
    ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
])

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

def barplot_mean_std(df_plot, title, outpath):
    fig, ax = plt.subplots(figsize=(10.5, 5))
    labels = ["\n".join(textwrap.wrap(m, width=18)) for m in df_plot["method"]]
    ax.bar(labels, df_plot["mean"], yerr=df_plot["std"], capsize=6)
    ax.set_ylabel("Accuracy (mean ± std)")
    ax.set_title(title)
    ax.tick_params(axis="x", labelsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# Run all CV methods
# -----------------------------
summaries = []

# 1) K-Fold (k=5)
start = time.time()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf_scores = cross_val_score(clf_pipe, X, y, cv=kf, scoring="accuracy")
summaries.append(summarize_scores("KFold (k=5)", kf_scores, start))

# 2) Stratified K-Fold (k=5)
start = time.time()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf_scores = cross_val_score(clf_pipe, X, y, cv=skf, scoring="accuracy")
summaries.append(summarize_scores("StratifiedKFold (k=5)", skf_scores, start))

# 3) Group K-Fold (k= min(5, n_groups)) using REAL groups = island
n_groups = groups.nunique()
gkf_splits = min(5, n_groups)
start = time.time()
gkf = GroupKFold(n_splits=gkf_splits)
gkf_scores = cross_val_score(clf_pipe, X, y, cv=gkf, groups=groups, scoring="accuracy")
summaries.append(summarize_scores(f"GroupKFold (k={gkf_splits}, groups=island)", gkf_scores, start))

# 4) LOOCV
start = time.time()
loo = LeaveOneOut()
loo_scores = cross_val_score(clf_pipe, X, y, cv=loo, scoring="accuracy")
summaries.append(summarize_scores("LOOCV", loo_scores, start))

# 5) Leave-P-Out (p=2) — limited subsamples for speed
start = time.time()
lpo = LeavePOut(p=2)
scores = []
for i, (tr_idx, te_idx) in enumerate(lpo.split(X, y)):
    if i >= 200:  # cap for runtime; raise if you want more
        break
    clf_pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    scores.append(clf_pipe.score(X.iloc[te_idx], y.iloc[te_idx]))
lpo_scores = np.array(scores)
summaries.append(summarize_scores("LeavePOut (p=2, 200 subsamples)", lpo_scores, start))

# -----------------------------
# Results
# -----------------------------
summary_df = pd.DataFrame(summaries)
print("\n=== Cross-validation comparison (Palmer Penguins) ===")
print(summary_df.round(4))

barplot_mean_std(
    summary_df,
    f"CV Methods Comparison — Penguins (target=species, groups=island; GroupKFold k={gkf_splits})",
    "cv_methods_comparison_penguins.png"
)
print("\nSaved plot -> cv_methods_comparison_penguins.png")

summary_df.round(6).to_csv("cv_methods_comparison_penguins.csv", index=False)
print("Saved table -> cv_methods_comparison_penguins.csv")
