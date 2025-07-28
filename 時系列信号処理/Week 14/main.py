import numpy as np
import matplotlib.pyplot as plt

# parameters
T = 50               # time steps
Q = 1.0              # process noise variance
R = 1.0              # observation noise variance
N = 100              # EnKF ensemble size

# reproducibility
np.random.seed(0)

# x_k = x_{k-1} + w,  y_k = x_k + v
x_true = np.zeros(T)
y_obs  = np.zeros(T)

# set initial state, x_0 = 0.0
x_true[0] = 0.0
y_obs[0]  = x_true[0] + np.random.normal(0.0, np.sqrt(R))

for k in range(1, T):
    x_true[k] = x_true[k-1] + np.random.normal(0.0, np.sqrt(Q))
    y_obs[k]  = x_true[k]   + np.random.normal(0.0, np.sqrt(R))

# 1 Initialization: x0^(i) ~ N(x0_mean, x0_var)
x0_mean, x0_var = 0.0, 1.0
ensemble = np.random.normal(x0_mean, np.sqrt(x0_var), size=N)

# EnKF main loop
enkf_mean = np.zeros(T)
for k in range(T):
    # 2 Time update (Prediction)
    # x_{k|k-1}^{(i)} = f(x_{k-1|k-1}^{(i)}) + w^{(i)}, where f(x)=x
    w_samp = np.random.normal(0.0, np.sqrt(Q), size=N)
    x_pred = ensemble + w_samp
    # y_{k|k-1}^{(i)} = h(x_{k|k-1}^{(i)}), where h(x)=x
    y_pred = x_pred.copy()

    # 3 Ensemble mean and covariance
    # x_bar = (1/N) * sum[x_i]
    x_bar = np.mean(x_pred)
    # y_bar = (1/N) * sum[y_i]
    y_bar = np.mean(y_pred)
    # P_xy = (1/(N-1)) * sum[(x_i - x_bar)*(y_i - y_bar)]
    P_xy = np.sum((x_pred - x_bar) * (y_pred - y_bar)) / (N - 1)
    # P_yy = (1/(N-1)) * sum[(y_i - y_bar)^2] + R
    P_yy = np.sum((y_pred - y_bar) ** 2) / (N - 1) + R

    # 4 Compute the Kalman gain
    # K_k = P_xy * inv(P_yy)
    K = P_xy / P_yy

    # 5 Update (Measurement update)
    # v^{(i)} ~ N(0,R)
    v_perturb = np.random.normal(0.0, np.sqrt(R), size=N)
    innov = (y_obs[k] + v_perturb) - y_pred
    ensemble = x_pred + K * innov

    enkf_mean[k] = np.mean(ensemble)

# plot
plt.figure(figsize=(10, 5))
plt.plot(x_true, label="True state")
plt.plot(y_obs,  label="Observation", linestyle=":", marker="o", markersize=3)
plt.plot(enkf_mean, label="EnKF estimate", linewidth=2)
plt.xlabel("Time step k")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()