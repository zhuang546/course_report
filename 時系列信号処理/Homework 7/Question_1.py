import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("hw7.csv", header=None)
yn = df.values.flatten()
N = len(yn)
fs = 1

# compute periodogram
n = np.arange(N)
frequencies = np.linspace(0, fs / 2, N // 2)
periodogram_values = []

for f in frequencies:
    omega = 2 * np.pi * f
    exponential = np.exp(-1j * omega * n)
    Xf = np.dot(yn, exponential)
    power = np.abs(Xf) ** 2 / N
    periodogram_values.append(power)

# plot
plt.figure(figsize=(8, 4))
plt.plot(frequencies, 10 * np.log10(periodogram_values))
plt.title("Periodogram of $y_n$")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (dB/Hz)")
plt.grid(True)
plt.tight_layout()
plt.show()