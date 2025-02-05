import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the Lotka-Volterra predator-prey equations
def lotka_volterra(t, z, alpha, beta, gamma, delta):
    x, y = z
    dxdt = alpha * x - beta * x * y
    dydt = -gamma * y + delta * x * y
    return [dxdt, dydt]

# System parameters
alpha = 1.0   # Prey growth rate
beta = 0.5    # Predation rate
gamma = 1.0   # Predator death rate
delta = 0.5   # Predator reproduction rate

# Time span for the simulation
t_span = (0, 50)
t_eval = np.linspace(*t_span, 1000)
# t_eval = np.linspace(*t_span, 5000) # High-resolution time points


# Initial conditions (various points in the phase space)
initial_conditions = [
    (2, 1), (3, 2), (4, 3), (5, 3), (6, 4)
]

# Solve the system for each initial condition
trajectories = []
for x0, y0 in initial_conditions:
    # sol = solve_ivp(lotka_volterra, t_span, [x0, y0], args=(alpha, beta, gamma, delta), t_eval=t_eval)
    sol = solve_ivp(lotka_volterra, t_span, [x0, y0], args=(alpha, beta, gamma, delta), method='DOP853', t_eval=t_eval) # DOP853 is a high-precision method
    trajectories.append(sol)

# Plot the level sets of the Lyapunov function for more clarity
x_vals = np.linspace(0.1, 10, 100)
y_vals = np.linspace(0.1, 10, 100)
X, Y = np.meshgrid(x_vals, y_vals)
V = -delta * X - beta * Y + gamma * np.log(X) + alpha * np.log(Y)

plt.figure(figsize=(8, 6)) # Larger figure size
contour = plt.contour(X, Y, V, levels=20, colors='gray', alpha=0.6, linestyles="dotted")
plt.clabel(contour, inline=True, fontsize=8, fmt="%.2f")
plt.plot([], [], color="gray", linestyle="dotted", label="Lyapunov Level Sets")  # Legend label

# Plot the equilibrium point
x_star = gamma / delta
y_star = alpha / beta
plt.scatter([x_star], [y_star], color="blue", label="Equilibrium $(x^*, y^*)$", s=100)

# Plot trajectories
for sol in trajectories:
    plt.plot(sol.y[0], sol.y[1], label=f"Init: ({sol.y[0,0]:.1f}, {sol.y[1,0]:.1f})")

# Set and show the plot
plt.xlabel("Prey Population (x)")
plt.ylabel("Predator Population (y)")
plt.title("2-Dimensional Lotka-Volterra Predator/Prey Dynamics")
plt.legend()
plt.grid(True)

plt.show()
