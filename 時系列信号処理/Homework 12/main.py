import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Config
n_steps = 20
Q = 0.01
R = 0.1
x_true = np.zeros(n_steps + 1)
y_obs = np.zeros(n_steps + 1)
x_est = np.zeros(n_steps + 1)
P = np.zeros(n_steps + 1)

# Initial conditions
x_true[0] = 1.0
x_est[0] = 1.0
P[0] = 0.2

# Functions
def f(x): return x + 0.1 * np.sin(x)
def h(x): return np.sin(x)
def F_jacobian(x): return 1.0 + 0.1 * np.cos(x)
def H_jacobian(x): return np.cos(x)

# Simulation
for k in range(1, n_steps + 1):
    # True state
    w_km1 = np.random.normal(0, np.sqrt(Q))
    x_true[k] = f(x_true[k - 1]) + w_km1

    # Noisy Observations
    v_k = np.random.normal(0, np.sqrt(R))
    y_obs[k] = h(x_true[k]) + v_k

    # EKF Prediction
    x_pred = f(x_est[k - 1])
    F_km1 = F_jacobian(x_est[k - 1])
    P_pred = F_km1 * P[k - 1] * F_km1 + Q

    # EKF Update
    H_k = H_jacobian(x_pred)
    y_pred = h(x_pred)
    K_k = P_pred * H_k / (H_k * H_k * P_pred + R)
    x_est[k] = x_pred + K_k * (y_obs[k] - y_pred)
    P[k] = (1 - K_k * H_k) * P_pred

# Plot
plt.figure(figsize=(8, 4))
plt.plot(range(n_steps + 1), x_true, label="True state $x_k$")
plt.scatter(range(n_steps + 1), y_obs, marker='o', label="Noisy Observations $y_k$")
plt.plot(range(n_steps + 1), x_est, linestyle='--', label="EKF estimate $\\hat{x}_{k|k}$")
plt.xlabel("Time step $k$")
plt.ylabel("Value")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()