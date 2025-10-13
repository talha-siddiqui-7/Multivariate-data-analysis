# If running in Colab, uncomment:
# !pip -q install imbalanced-learn==0.12.3 scikit-learn>=1.3 matplotlib

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, average_precision_score, confusion_matrix)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

import matplotlib.pyplot as plt

RNG = 7

# -----------------------------
# 1) Synthetic imbalanced data
# -----------------------------
X, y = make_classification(n_samples=20000, n_features=20, n_informative=6, n_redundant=4,
                           n_clusters_per_class=2, weights=[0.99, 0.01], flip_y=0.005,
                           class_sep=1.2, random_state=RNG)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RNG
)

def evaluate(clf, X_tr, y_tr, X_te, y_te, name, threshold=None):
    """Compute a compact metric table. Supports threshold tuning if predict_proba is available."""
    clf.fit(X_tr, y_tr)
    y_score = None
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_te)[:, 1]
    elif hasattr(clf, "decision_function"):
        # Map decision function to [0,1] via rank-based scaling (keeps ordering for AUCs)
        s = clf.decision_function(X_te)
        r = (s - s.min()) / (s.max() - s.min() + 1e-12)
        y_score = r
    else:
        y_score = None

    if threshold is not None and y_score is not None:
        y_pred = (y_score >= threshold).astype(int)
    else:
        y_pred = clf.predict(X_te)

    acc = accuracy_score(y_te, y_pred)
    bacc = balanced_accuracy_score(y_te, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="binary", zero_division=0)
    roc = roc_auc_score(y_te, y_score) if y_score is not None else np.nan
    ap = average_precision_score(y_te, y_score) if y_score is not None else np.nan
    cm = confusion_matrix(y_te, y_pred)

    return {
        "model": name,
        "threshold": threshold,
        "accuracy": acc,
        "balanced_acc": bacc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "roc_auc": roc,
        "pr_auc": ap,
        "cm": cm,
    }

def print_row(res):
    print(f"{res['model']:<28} "
          f"acc={res['accuracy']:.3f}  bal_acc={res['balanced_acc']:.3f}  "
          f"P={res['precision']:.3f}  R={res['recall']:.3f}  F1={res['f1']:.3f}  "
          f"ROC-AUC={res['roc_auc']:.3f}  PR-AUC={res['pr_auc']:.3f}  thr={res['threshold']}")
    print(f"  CM:\n{res['cm']}")

# ------------------------------------------
# 2) Baselines (illustrate the "accuracy trap")
# ------------------------------------------
print("=== Baselines on imbalanced data ===")
baselines = [
    ("LDA (default)", LinearDiscriminantAnalysis()),
    ("DecisionTree", DecisionTreeClassifier(random_state=RNG)),
    ("RandomForest", RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1)),
    ("SVM rbf", SVC(kernel="rbf", probability=True, class_weight=None, random_state=RNG)),
]
for name, clf in baselines:
    res = evaluate(clf, X_train, y_train, X_test, y_test, name)
    print_row(res)

# ----------------------------------------------------------
# 3) Algorithm-level: class weights / cost-sensitive learning
#    (Tree, RF, SVM support class_weight='balanced')
# ----------------------------------------------------------
print("\n=== Class-weighted models (no resampling) ===")
weighted = [
    ("DecisionTree (weighted)", DecisionTreeClassifier(class_weight="balanced", random_state=RNG)),
    ("RandomForest (weighted)", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                      random_state=RNG, n_jobs=-1)),
    ("SVM rbf (weighted)", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RNG)),
]
for name, clf in weighted:
    res = evaluate(clf, X_train, y_train, X_test, y_test, name)
    print_row(res)

# ----------------------------------------------------------
# 4) Data-level: resampling (leakage-safe with imblearn.Pipeline)
# ----------------------------------------------------------
print("\n=== Resampling pipelines ===")
resamplers = [
    ("RandomOver + LDA", Pipeline([("ros", RandomOverSampler(random_state=RNG)),
                                   ("clf", LinearDiscriminantAnalysis())])),
    ("SMOTE + LDA", Pipeline([("smote", SMOTE(random_state=RNG)),
                              ("clf", LinearDiscriminantAnalysis())])),
    ("RandomUnder + Tree", Pipeline([("rus", RandomUnderSampler(random_state=RNG)),
                                     ("clf", DecisionTreeClassifier(random_state=RNG))])),
    ("SMOTE + RF", Pipeline([("smote", SMOTE(random_state=RNG)),
                             ("clf", RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1))])),
    ("SMOTE + SVM rbf", Pipeline([("smote", SMOTE(random_state=RNG)),
                                  ("clf", SVC(kernel="rbf", probability=True, random_state=RNG))]))
]
for name, pipe in resamplers:
    res = evaluate(pipe, X_train, y_train, X_test, y_test, name)
    print_row(res)

# -------------------------------------------------------------------
# 5) Threshold moving (optimize F-beta on validation -> apply on test)
#    Here: use LDA (probabilistic) and SVM (probability=True)
# -------------------------------------------------------------------
from sklearn.metrics import precision_recall_curve

def find_best_threshold_for_fbeta(y_true, scores, beta=2.0):
    # F_beta: (1+β^2) * P*R / (β^2*P + R)
    P, R, T = precision_recall_curve(y_true, scores)
    fbeta = (1+beta**2)*P*R / (beta**2*P + R + 1e-12)
    idx = np.nanargmax(fbeta)
    return T[idx if idx < len(T) else -1], P[idx], R[idx], fbeta[idx]

print("\n=== Threshold tuning on validation split (optimize F2) ===")
# Make a small validation split from the train set to tune threshold (avoid test leakage)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                            stratify=y_train, random_state=RNG)

def tune_and_eval_threshold(clf, name):
    clf.fit(X_tr, y_tr)
    if hasattr(clf, "predict_proba"):
        val_score = clf.predict_proba(X_val)[:, 1]
        test_score = clf.predict_proba(X_test)[:, 1]
    else:
        val_score = clf.decision_function(X_val)
        s = clf.decision_function(X_test)
        # map to [0,1] for PR metrics
        test_score = (s - s.min()) / (s.max() - s.min() + 1e-12)

    thr, p_, r_, f2_ = find_best_threshold_for_fbeta(y_val, val_score, beta=2.0)
    res = evaluate(clf, X_train, y_train, X_test, y_test, f"{name} (thr-tuned F2)", threshold=thr)
    print_row(res)
    print(f"  Tuned threshold={thr:.3f}  (val: P={p_:.3f}, R={r_:.3f}, F2={f2_:.3f})")

tune_and_eval_threshold(LinearDiscriminantAnalysis(), "LDA")
tune_and_eval_threshold(SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RNG), "SVM rbf weighted")

# -----------------------------------------
# 6) (Optional) Precision-Recall curve plot
# -----------------------------------------
def plot_pr_curve(clf, X_te, y_te, label):
    clf.fit(X_train, y_train)
    if hasattr(clf, "predict_proba"):
        s = clf.predict_proba(X_te)[:, 1]
    else:
        s = clf.decision_function(X_te)
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    P, R, _ = precision_recall_curve(y_te, s)
    ap = average_precision_score(y_te, s)
    plt.plot(R, P, lw=2, label=f"{label} (AP={ap:.3f})")

plt.figure(figsize=(6,5))
plot_pr_curve(LinearDiscriminantAnalysis(), X_test, y_test, "LDA")
plot_pr_curve(Pipeline([("smote", SMOTE(random_state=RNG)), ("clf", LinearDiscriminantAnalysis())]),
              X_test, y_test, "SMOTE + LDA")
plot_pr_curve(SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RNG),
              X_test, y_test, "SVM rbf (weighted)")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall (minority class)")
plt.legend(); plt.grid(True); plt.show()
