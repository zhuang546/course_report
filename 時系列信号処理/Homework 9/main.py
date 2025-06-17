import numpy as np
import matplotlib.pyplot as plt
# state-space model simulation

A= np.array([[0.9, 0.1], [0, 0.95]])
B = np.array([[0.1], [0.05]])
x0 = np.array([[1], [0]])
uk = np.array([[1]])
num_steps = 30

def update(xk):
    omega = np.random.normal(0, 0.1, (2, 1))
    return A @ xk + B * uk + omega

def simulate():
    xk_list = np.zeros((num_steps+1, 2, 1))
    xk_list[0] = x0
    for k in range(num_steps):
        xk_list[k+1] = update(xk_list[k])
    return xk_list

xk_list = simulate()
# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(xk_list[:, 0, 0], label='x1')
plt.plot(xk_list[:, 1, 0], label='x2')
plt.title('State-Space Model Simulation')
plt.xlabel('k')
plt.ylabel('State Values')
plt.legend()
plt.grid()
plt.show()