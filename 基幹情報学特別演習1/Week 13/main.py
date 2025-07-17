import numpy as np
import matplotlib.pyplot as plt

def fcm0(X, C, theta):
    # Visualize the first 5 iterations of Fuzzy C‑Means clustering.

    plt.figure(figsize=(30, 5))
    n, m = X.shape

    # initial random membership matrix
    d_rand = np.random.rand(C, n)
    u = d_rand / d_rand.sum(axis=0, keepdims=True)
    expo = 2.0 / (theta - 1)

    for step in range(5):
        # 1. update centroids
        u_theta = u ** theta
        b = (u_theta @ X) / u_theta.sum(axis=1, keepdims=True)

        # 2. distance matrix (C × n)
        d = np.linalg.norm(b[:, None, :] - X[None, :, :], axis=2)

        # 3. update memberships
        zero_mask = (d == 0)
        u_new = np.zeros_like(d)

        # columns where some distance is zero → crisp assignment
        zero_cols = zero_mask.any(axis=0)
        u_new[zero_mask] = 1

        if np.any(~zero_cols):
            d_non = d[:, ~zero_cols]
            w = d_non ** (-expo)
            u_new[:, ~zero_cols] = w / w.sum(axis=0, keepdims=True)

        u = u_new

        # 4. objective function
        obj = ((u ** theta) * (d ** 2)).sum()

        # 5. plot this step
        plt.subplot(1, 5, step + 1)
        plt.scatter(X[:, 0], X[:, 1], color=u.T, edgecolors='k')
        plt.scatter(b[:, 0], b[:, 1], marker='x', c='k', s=150)
        plt.title(f"Step {step}\nθ={theta}, J={obj:.1f}", fontsize=18)

    plt.show()

# main
np.random.seed(0)
X = np.vstack(
    (
        np.random.randn(100, 2) + (0, 0),
        np.random.randn(100, 2) + (3, 5),
        np.random.randn(100, 2) + (6, 0),
    )
)

for theta in [1.05, 2.0, 5.0, 10.0]:
    fcm0(X, C=3, theta=theta)



