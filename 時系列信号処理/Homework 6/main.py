import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

a = 0.75
sigma_v2 = 1
sigma_w2 = 0.25
N = 2

y = pd.read_csv("hw6.csv", header=None).values.flatten()

# Compute R_xx(τ)
def R_xx(tau):
    return (16 / 7) * (a ** abs(tau))

# Construct autocorrelation matrix R_yy and cross-correlation vector R_yx
def get_Ryy_and_Ryx(n, predict=False):
    lags = np.arange(-N, 1)[::-1]  # [-2, -1, 0]
    Ryx = np.array([R_xx(tau + 1) if predict else R_xx(tau) for tau in lags])  # shape: (3,)
    
    Ryy = np.zeros((N+1, N+1))
    for i in range(N+1):
        for j in range(N+1):
            Ryy[i, j] = R_xx(i - j) + (sigma_w2 if i == j else 0)
    return Ryx, Ryy

# Run LMMSE estimation for n = 10 to 100
n_values = np.arange(10, 101)
xhat_n_given = []
xhat_nplus1_given = []

for n in n_values:
    y_obs = y[n - N: n + 1][::-1]

    # Estimate x_n
    Ryx_filt, Ryy = get_Ryy_and_Ryx(n, predict=False)
    xhat_n = Ryx_filt @ np.linalg.inv(Ryy) @ y_obs
    xhat_n_given.append(xhat_n)

    # Estimate x_{n+1}
    Ryx_pred, _ = get_Ryy_and_Ryx(n, predict=True)
    xhat_n1 = Ryx_pred @ np.linalg.inv(Ryy) @ y_obs
    xhat_nplus1_given.append(xhat_n1)

# Plot
plt.figure(figsize=(12, 5))
plt.plot(n_values, xhat_n_given, label=r'$\hat{x}_n|y_{n-2:n}$ (smoothing)')
plt.title('LMMSE Estimate of $x_n$ using $y_{n-2:n}$')
plt.xlabel('n')
plt.ylabel(r'$\hat{x}_n$')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(n_values, xhat_nplus1_given, label=r'$\hat{x}_{n+1}|y_{n-2:n}$ (prediction)', color='orange')
plt.title('LMMSE Prediction of $x_{n+1}$ using $y_{n-2:n}$')
plt.xlabel('n')
plt.ylabel(r'$\hat{x}_{n+1}$')
plt.grid(True)
plt.legend()
plt.show()