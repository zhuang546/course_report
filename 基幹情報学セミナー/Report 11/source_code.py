import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 50  # Number of paths
f_c = 2e9  # Carrier frequency
v = 60  # Velocity
c = 3e8  # Speed of light
f_D = (v / c) * f_c  # Maximum Doppler frequency
T_s = 1e-4  # Sampling period
f_D_bar = f_D * T_s  # Normalized Maximum Doppler frequency
duration = 0.04  # Duration
num_samples = int(duration / T_s)  # Total number of samples
n = np.arange(num_samples)  # Time indices

# Path parameters
np.random.seed(1)  # For reproducibility
theta = np.random.uniform(0, 2 * np.pi, N)  # Uniformly distributed angles
a = np.random.randn(N) + 1j * np.random.randn(N)  # Reflecting coefficients (independent Gaussian)

# Compute time-selective channel
h_n = np.zeros(num_samples, dtype=complex)
for i in range(N):
    h_n += a[i] * np.exp(1j * 2 * np.pi * f_D_bar * np.cos(theta[i]) * n)
h_n /= np.sqrt(N)

# Plot the amplitude of the channel
plt.figure(figsize=(10, 6))
plt.plot(n * T_s, np.abs(h_n), linewidth=0.8)
plt.title("Amplitude of channel")
plt.xlabel("time (seconds)")
plt.ylabel("Envelope of the fading coefficient")
plt.show()