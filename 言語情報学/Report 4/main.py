from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Simulate data
centers = [[-5, -2], [0, 0], [3, 5], [8, -3]]
cluster_std = [1.0, 2.5, 1.0, 1.8]  # different standard deviations for each cluster
X, y_true = make_blobs(n_samples=500, centers=centers, cluster_std=cluster_std, random_state=42)

# use EM（GMM）to cluster the data to 4 clusters
gmm = GaussianMixture(n_components=4, random_state=42)
gmm.fit(X)
labels = gmm.predict(X)

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
# left：True Labels
ax[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10', s=30)
ax[0].set_title("True Labels (Simulated Clusters)")
# right：GMM Clustering Result
ax[1].scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=30)
ax[1].set_title("GMM Clustering (EM) Result")

plt.tight_layout()
plt.show()