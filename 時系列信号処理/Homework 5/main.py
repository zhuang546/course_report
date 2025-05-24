import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt

y = np.loadtxt("hw5.csv")

def sample_auto_correlation(y, l):
    N = len(y)
    return np.sum(y[m]*y[m-l] for m in range(l, N)) / N

def estimate_ar_parameters(y, order):
    # estimate R(0) to R(N)
    R = np.array([sample_auto_correlation(y, lag) for lag in range(order + 1)])
    # the matrix and the vector in the Yule-Walker equations
    R_matrix = np.array([[R[abs(i - j)] for j in range(order)] for i in range(order)])
    a_vector = R[1:order + 1]

    # solve Yule-Walker equations
    a = solve(R_matrix, a_vector)
    sigma_w2 = R[0] - np.dot(a, a_vector)
    return a, sigma_w2

orders = range(1, 6)
coeffs_all = []
noise_vars = []

# obtain AR(N) models with Yule-Walker equations
for N in orders:
    a, sigma2 = estimate_ar_parameters(y, N)
    coeffs_all.append(a)
    noise_vars.append(sigma2)
    print(f"AR({N}) coefficients: {a}")
    print(f"variance σ_w^2: {sigma2:.6f}\n")

# plot the 
plt.figure(figsize=(8, 4))
plt.plot(orders, noise_vars, marker='o')
plt.xlabel("N")
plt.ylabel("σ_w²")
plt.xticks(orders)
plt.grid(True)
plt.tight_layout()
plt.show()