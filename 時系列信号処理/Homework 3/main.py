import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# read data from CSV file
data = pd.read_csv('hw3.csv', header=None)
x = data.iloc[:, 0].values
y = data.iloc[:, 1].values
N = len(x)

# set up the matrix H
H = np.column_stack((np.ones(N), x, x**2))  # H = [1, x, x^2]

# compute the ML estimates
theta_hat = np.linalg.inv(H.T @ H) @ (H.T @ y)
theta0_hat, theta1_hat, theta2_hat = theta_hat

# print the estimates
print(f"θ_0 = {theta0_hat:.4f}, θ_1 = {theta1_hat:.4f}, θ_2 = {theta2_hat:.4f}")

# plot the data points and the regression curve
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='red', label='Data Points')

x_curve = np.linspace(min(x), max(x), 300)  
y_curve = theta0_hat + theta1_hat * x_curve + theta2_hat * x_curve**2
plt.plot(x_curve, y_curve, color='blue', label='Regression Curve')

# plot settings
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()