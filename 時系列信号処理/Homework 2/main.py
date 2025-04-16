from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt

def F(alpha):
    integrand = lambda x: x**(alpha - 1) * np.exp(-x)
    result = quad(integrand, 0, np.inf)[0]
    return result

def Beta(theta, a, b):
    temp = F(a + b) / (F(a) * F(b))
    result = temp * (theta**(a - 1)) * ((1 - theta)**(b - 1))
    return result

a = 2
b = 3

plt.figure(figsize=(8, 6))
x_plot = np.linspace(0, 1, 100)
plt.plot(x_plot, Beta(x_plot, a, b), 'r', label=f'θ~Beta({a}, {b})')

plt.xlabel("θ")
plt.ylabel("p(θ)")
plt.title("Beta Distribution")

plt.legend()
plt.grid(True)
plt.show()