import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('TrainingDataForReport5.csv')

x = data['x'].values 
y = data['y'].values
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Data points')
Knots_K = 4
c_ks = np.array([2, 4, 6, 8])

def h_0(x):
    return np.ones_like(x)
def h_1(x):
    return x
def h_2(x):
    return x**2
def h_3(x):
    return x**3
def h_k_3(x, c_k):
    return np.maximum(0, (x - c_k)**3)

def get_X(x_training):
    X = np.zeros((len(x_training), Knots_K+4))
    X[:, 0] = h_0(x_training)
    X[:, 1] = h_1(x_training)
    X[:, 2] = h_2(x_training)
    X[:, 3] = h_3(x_training)
    for k in range(1, Knots_K + 1):
        X[:, k+3] = h_k_3(x_training, c_ks[k-1])
    return X

def get_H(X):
    return np.linalg.inv(X.T @ X) @ X.T

def get_bata_hat(x_training, y_training):
    X = get_X(x_training)
    H = get_H(X)
    return H @ y_training

def train():
    beta_hat = get_bata_hat(x, y)
    x_plot = np.linspace(np.min(x), np.max(x), 100)
    print(f'training beta_hat : {beta_hat}')
    plt.plot(x_plot, get_X(x_plot) @ beta_hat,color='r', label=f'Knots_K={Knots_K}')

def CV_LOO_magic():
    X = get_X(x)
    H = get_H(X)
    beta_hat = H @ y
    f_hat = X @ beta_hat
    h_ii = np.diag(X @ H)
    return np.mean(((y - f_hat)/(1-h_ii))**2)

def CV_LOO_no_magic():
    error = np.zeros(len(x))
    for k in range(len(x)):
        mask = np.zeros(len(x), dtype=bool)
        mask[k] = True
        beta_hat = get_bata_hat(x[~mask], y[~mask])
        f_hat = get_X(x[mask]) @ beta_hat
        error[mask] = (y[mask] - f_hat)**2
    return np.mean(error)

train()
print(f'CV_LOO_magic: {CV_LOO_magic()}')
print(f'CV_LOO_no_magic: {CV_LOO_no_magic()}')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Knots, K = {Knots_K}')
plt.legend()
plt.show()