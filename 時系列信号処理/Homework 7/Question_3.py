import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("hw7.csv", header=None)
yn = df.values.flatten()
N = len(yn)

# Step 1: estimate R̂xx(τ)，τ = 0, 1, 2
def estimate_autocorrelation(x, max_lag):
    R = []
    x = x - np.mean(x)
    for tau in range(max_lag + 1):
        r = np.sum(x[tau:] * x[:N - tau]) / (N - tau)
        R.append(r)
    return np.array(R)

R_hat = estimate_autocorrelation(yn, max_lag=2)

# Step 2: solve Yule-Walker equations for AR(2)
# R(1) = -a1*R(0) - a2*R(1)
# R(2) = -a1*R(1) - a2*R(0)
R0, R1, R2 = R_hat[0], R_hat[1], R_hat[2]
R_matrix = np.array([[R0, R1],
                     [R1, R0]])
r_vector = -np.array([R1, R2])
a = np.linalg.solve(R_matrix, r_vector)
a1, a2 = a

# estimate σ_w^2
sigma_w2 = R0 + a1 * R1 + a2 * R2

# Step 3: construct PSD using the Yule-Walker method
# S(ω) = σ_w^2 / |1 + a1 e^{-jω} + a2 e^{-j2ω}|^2
n_fft = 1024
frequencies = np.linspace(0, 0.5, n_fft // 2)
omega = 2 * np.pi * frequencies

# Calculate the PSD
numerator = sigma_w2
denominator = np.abs(1 + a1 * np.exp(-1j * omega) + a2 * np.exp(-2j * omega))**2
S_yw = numerator / denominator

# Step 4: plot
plt.figure(figsize=(8, 4))
plt.plot(frequencies, 10 * np.log10(S_yw))
plt.title("Yule-Walker PSD Estimate (AR(2))")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (dB/Hz)")
plt.grid(True)
plt.tight_layout()
plt.show()