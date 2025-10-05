# The penguins dataset (from seaborn) contains measurements for three penguin species:
# Adelie, Chinstrap, and Gentoo. It includes features such as bill length, bill depth,
# flipper length, and body mass, collected from Palmer Archipelago in Antarctica.
# This dataset is often used for demonstrating clustering, visualization, and classification tasks.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import umap

# --- Load & preprocess the penguins dataset
penguins = sns.load_dataset("penguins")
# Keep only numeric features to avoid metric mismatches
num_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
df = penguins[num_cols + ["species"]].dropna().reset_index(drop=True)

X = df[num_cols].values
y = df["species"].values

# Scale features (UMAP works without scaling, but scaling helps metrics like cosine/manhattan)
X_scaled = StandardScaler().fit_transform(X)

# --- Utility: plot a 2D UMAP embedding
def plot_umap_2d(X, y, title="", **umap_kwargs):
    reducer = umap.UMAP(random_state=42, **umap_kwargs)
    emb = reducer.fit_transform(X)
    # Optional: a quick quantitative signal for clustering (silhouette on embedded space)
    try:
        labels = pd.factorize(y)[0]
        sil = silhouette_score(emb, labels)
        title = f"{title}\nSilhouette (embedded): {sil:.2f}"
    except Exception:
        pass
    # Plot
    plt.scatter(emb[:,0], emb[:,1], s=12, c=pd.factorize(y)[0], cmap="tab10", alpha=0.9)
    plt.title(title, fontsize=11)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.xticks([])
    plt.yticks([])

# --- 1) Effect of n_neighbors
nn_values = [5, 15, 50, 100]
plt.figure(figsize=(12, 8))
for i, nn in enumerate(nn_values, 1):
    plt.subplot(2, 2, i)
    plot_umap_2d(
        X_scaled, y,
        title=f"n_neighbors = {nn} (min_dist=0.1, metric='euclidean')",
        n_neighbors=nn, min_dist=0.1, metric="euclidean", n_components=2
    )
plt.tight_layout()
plt.show()

# --- 2) Effect of min_dist
md_values = [0.001, 0.1, 0.5]
plt.figure(figsize=(12, 4))
for i, md in enumerate(md_values, 1):
    plt.subplot(1, 3, i)
    plot_umap_2d(
        X_scaled, y,
        title=f"min_dist = {md} (n_neighbors=15, metric='euclidean')",
        n_neighbors=15, min_dist=md, metric="euclidean", n_components=2
    )
plt.tight_layout()
plt.show()

# --- 3) Effect of metric
metrics = ["euclidean", "cosine", "manhattan"]
plt.figure(figsize=(12, 4))
for i, m in enumerate(metrics, 1):
    plt.subplot(1, 3, i)
    plot_umap_2d(
        X_scaled, y,
        title=f"metric = '{m}' (n_neighbors=15, min_dist=0.1)",
        n_neighbors=15, min_dist=0.1, metric=m, n_components=2
    )
plt.tight_layout()
plt.show()

# --- 4) (Optional) n_components = 3 demonstration
# Note: 3D scatter; useful if you want to pass this to a downstream clustering step in 3D.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

reducer_3d = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean",
                       n_components=3, random_state=42)
emb3 = reducer_3d.fit_transform(X_scaled)

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(emb3[:,0], emb3[:,1], emb3[:,2], s=10, c=pd.factorize(y)[0], cmap="tab10", alpha=0.9)
ax.set_title("UMAP 3D (n_components=3)")
ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); ax.set_zlabel("UMAP-3")
plt.show()

# --- Quick takeaway prints for your report
print("Quick tuning takeaways:")
print("- Smaller n_neighbors → emphasizes local structure (tends to create more separated, smaller clusters).")
print("- Larger n_neighbors → emphasizes global structure (clusters may merge but global gradients become clearer).")
print("- Smaller min_dist → tighter, denser clusters; larger min_dist → more spread out.")
print("- Metric matters: try euclidean (default), cosine (directional/sparse features), manhattan (L1).")
