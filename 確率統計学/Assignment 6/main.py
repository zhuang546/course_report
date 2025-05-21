import numpy as np
import matplotlib.pyplot as plt

# Diffusion Equation Numerical Solution
delta_t = 0.002
delta_x = 0.1
times = np.array([0, 0.05, 0.1, 0.2, 0.4])

def c_x0(x):
    if x > 0 and x < 0.5:
        return 2*x
    elif x >= 0.5 and x < 1:
        return 2*(1-x)
    else:
        return 0

x = np.linspace(0, 1, int(1/delta_x) + 1)
c_x_t = np.array([c_x0(xi) for xi in x])
time_steps = np.array([int(t / delta_t) for t in times])

plt.figure(figsize=(6, 6))
for step in range(0, max(time_steps) + 1):
    if step in time_steps:
        plt.plot(x, c_x_t, label=f't={step * delta_t}')
    c_temp  = c_x_t.copy()
    for i in range(1,len(x)-1):
        c_x_t[i] = c_temp[i] + delta_t/(delta_x**2)*(c_temp[i+1] - 2*c_temp[i] + c_temp[i-1])

plt.title('Diffusion Equation Numerical Solution')
plt.xlabel('x')
plt.ylabel('c(x,t)')
plt.xticks([0, 0.25, 0.5, 0.75, 1])
plt.yticks([0, 0.25, 0.5, 0.75, 1])
plt.grid()
plt.legend()
plt.show()