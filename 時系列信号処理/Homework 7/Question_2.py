import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 重新加载数据
df = pd.read_csv("hw7.csv", header=None)
yn = df.values.flatten()
N = len(yn)
fs = 1

# Step 1: R̂xx(τ)
def estimate_autocorrelation(x, max_lag):
    R = []
    x = x - np.mean(x)
    N = len(x)
    for tau in range(-max_lag, max_lag + 1):
        if tau >= 0:
            r = np.sum(x[tau:] * x[:N - tau]) / (N - tau)
        else:
            r = np.sum(x[:N + tau] * x[-tau:]) / (N + tau)
        R.append(r)
    return np.array(R)

max_lag = N // 2
lags = np.arange(-max_lag, max_lag + 1)
R_hat = estimate_autocorrelation(yn, max_lag)

# Step 2: Blackman window function
def blackman_window(M):
    n = np.arange(-M, M + 1)
    w = 0.42 - 0.5 * np.cos(2 * np.pi * (n + M) / (2 * M)) + 0.08 * np.cos(4 * np.pi * (n + M) / (2 * M))
    return w

w_blackman = blackman_window(max_lag)

# Step 3: R̂xx(τ) * w(τ)
R_win = R_hat * w_blackman

n_fft = 1024
frequencies = np.linspace(0, fs / 2, n_fft // 2)
omega = 2 * np.pi * frequencies

# compute DTFT
tau = np.arange(-max_lag, max_lag + 1)
S_BT = []
for w in omega:
    exp_term = np.exp(-1j * w * tau)
    S = np.sum(R_win * exp_term)
    S_BT.append(np.real(S))

# 转换为 dB/Hz 并绘图
plt.figure(figsize=(8, 4))
plt.plot(frequencies, 10 * np.log10(S_BT))
plt.title("Blackman-Tukey PSD Estimate")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (dB/Hz)")
plt.grid(True)
plt.tight_layout()
plt.show()
