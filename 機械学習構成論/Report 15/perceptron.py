import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def train_mlp(X, t, hidden_size=2, lr=0.1, iters=10000, seed=10, init_scale=0.5):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    # Initialize weights
    W1 = rng.normal(0, init_scale, size=(d, hidden_size))
    b1 = np.zeros((1, hidden_size))
    W2 = rng.normal(0, init_scale, size=(hidden_size, 1))
    b2 = np.zeros((1, 1))
    errors = []
    for _ in range(iters):
        # Forward pass
        h = sigmoid(X @ W1 + b1)
        y = sigmoid(h @ W2 + b2)
        # Sum of squared errors
        e = np.sum((t - y) ** 2)
        errors.append(e)
        # Backpropagation (batch)
        delta2 = 2 * (y - t) * y * (1 - y)
        dW2 = h.T @ delta2
        db2 = np.sum(delta2, axis=0, keepdims=True)
        delta1 = (delta2 @ W2.T) * h * (1 - h)
        dW1 = X.T @ delta1
        db1 = np.sum(delta1, axis=0, keepdims=True)
        # Update parameters (gradient descent)
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1
    return np.array(errors)

# data (label, x, y)
sep  = np.array([
    [0,0,1],
    [1,1,0]
], dtype=float)
insep = np.array([
    [1,0,1],
    [1,1,0],
    [0,0,0],
    [0,1,1]
], dtype=float)

# train
err_sep   = train_mlp(sep[:,1:], sep[:,[0]])
err_insep = train_mlp(insep[:,1:], insep[:,[0]])

# plot
plt.figure(); plt.plot(err_sep)
plt.title("Linearly Separable")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.tight_layout()
plt.figure(); plt.plot(err_insep)
plt.title("Linearly Inseparable / XOR")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.tight_layout()
plt.show()