import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# CLE simulation
k_1 = 1
k_2 = 0.1
dt = 0.01

steps = 100000
turns = 1000

X = np.full((turns, steps+1), 15.0)

def f(n):
    return k_1 - k_2 * X[:,n]
def g_1():
    return np.sqrt(k_1)
def g_2(n):
    return np.sqrt(k_2 * X[:,n])

for n in range(steps):
    xi_1 = np.array([np.random.normal(0, 1) for _ in range(turns)])
    xi_2 = np.array([np.random.normal(0, 1) for _ in range(turns)])
    X[:,n+1] = np.maximum(X[:,n] + f(n)*dt + g_1()*np.sqrt(dt)*xi_1 + g_2(n)*np.sqrt(dt)*xi_2, 0)

plt.hist(X[:,-1], bins=30, density=True, color='skyblue', edgecolor='black')
mean = np.mean(X[:,-1])
var = np.var(X[:,-1])
print()
print(f'Mean: {mean}, Variance: {var}')

x_plot = np.linspace(0, 20, 100)
plt.plot(x_plot, norm.pdf(x_plot, 10, np.sqrt(10)), color='red', label='Normal PDF')

plt.title('CLE Simulation Histogram')
plt.xlabel('X')
plt.ylabel('Density')
plt.grid()
plt.legend()
plt.show()

x_plot = np.linspace(0, int(steps*dt), steps+1)
plt.plot(x_plot, X[0,:], label='Sample Path 1')
plt.title('Sample Path of CLE Simulation')
plt.xlabel('Time')
plt.ylabel('X')
plt.xticks(np.arange(0, int(steps*dt)+1, steps*dt//5))
plt.yticks(np.arange(0, 20, 5))
plt.grid()
plt.legend()
plt.show()