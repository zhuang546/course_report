import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# generate data
np.random.seed(42)
T = 50
x_true = np.zeros(T)
y_obs = np.zeros(T)

for k in range(1, T):
    x_true[k] = x_true[k-1] + np.random.normal(0, 1)  # true state
y_obs = x_true + np.random.normal(0, 1, T)            # observations

# SIR filtering
N = 1000                              # particle num
particles = np.random.normal(0, 1, N) # initial particles
weights = np.ones(N) / N              # initial weights
x_hat = np.zeros(T)                   # filtered estimates

def systematic_resample(particles, weights):
    # Systematic resampling
    N = len(particles)
    positions = (np.arange(N) + np.random.uniform()) / N
    indexes = np.zeros(N, dtype=int)
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    particles[:] = particles[indexes]
    weights.fill(1.0 / N)

for k in range(T):
    # Prediction
    if k > 0:
        particles += np.random.normal(0, 1, N)
    # Update
    weights *= np.exp(-0.5 * (y_obs[k] - particles)**2)
    # Normalize
    weights /= np.sum(weights)
    # Filtering Estimate
    x_hat[k] = np.sum(weights * particles)
    # Resampling
    Neff = 1.0 / np.sum(weights**2)
    if Neff < N / 2:
        systematic_resample(particles, weights)

# plot
plt.figure(figsize=(8,4))
plt.plot(x_true, label='True state')
plt.scatter(np.arange(T), y_obs, s=10, label='Observations')
plt.plot(x_hat, label='SIR Filtering Estimate')
plt.xlabel('Time step k')
plt.ylabel('Value')
plt.legend()
plt.show()