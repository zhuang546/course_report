import numpy as np
import matplotlib.pyplot as plt

# Initial
n_steps = 50
A = np.array([[1, 1],
              [0, 1]])
H = np.array([[1, 0]])
Q = np.array([[0.1, 0],
              [0, 0.01]])
R = np.array([[0.5]])
x0 = np.array([0.0, 1.0])

xk_true = np.zeros((n_steps, 2))
xk_true[0] = x0

# Simulate
wk = np.random.multivariate_normal(mean=[0, 0], cov=Q, size=n_steps)
vk = np.random.normal(loc=0, scale=np.sqrt(R[0, 0]), size=n_steps)

yk = np.zeros(n_steps)
yk[0] = (H @ xk_true[0])[0] + vk[0]

for k in range(1, n_steps):
    xk_true[k] = A @ xk_true[k - 1] + wk[k - 1]
    yk[k] = (H @ xk_true[k])[0] + vk[k]

# Kalman filter initialization
x_hat = np.zeros((n_steps, 2))
P = np.zeros((n_steps, 2, 2))

x_hat[0] = np.array([0.0, 0.5])
P[0] = np.eye(2)

# Kalman filter recursion
for k in range(1, n_steps):
    # Prediction
    x_pred = A @ x_hat[k - 1]
    P_pred = A @ P[k - 1] @ A.T + Q

    # Update
    S = H @ P_pred @ H.T + R
    Kk = P_pred @ H.T @ np.linalg.inv(S)
    innovation = yk[k] - (H @ x_pred)[0]
    x_hat[k] = x_pred + (Kk.flatten() * innovation)
    P[k] = (np.eye(2) - Kk @ H) @ P_pred

# Compute RMSE
rmse = np.sqrt(np.mean((xk_true[:, 0] - x_hat[:, 0]) ** 2))
print(f"RMSE: {rmse:.4f}")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(range(n_steps), xk_true[:, 0], label='True Position')
plt.scatter(range(n_steps), yk, s=15, label='Noisy Observation')
plt.plot(range(n_steps), x_hat[:, 0], label='Kalman Estimate')
plt.xlabel('Time step')
plt.ylabel('Position')
plt.legend()
plt.grid(True)
plt.show()