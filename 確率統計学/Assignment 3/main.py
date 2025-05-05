import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

# generate an exponential random variable
def exp_rand_var(lam):
    return -np.log(np.random.uniform(0,1)) / lam

# obtain a Poisson-distributed variable
def poisson_rand_var(lam):
    sum = 0
    n = 0
    while sum < 1:
        sum += exp_rand_var(lam)
        n += 1
    return n - 1

# obtain a value of the theoretical Poisson distribution to compare
def poisson_pdf(k, lam):
    return (lam**k * np.exp(-lam)) / factorial(k)

lam = 4
sample_num = 10000

# generate random variables
poisson_rand_vars = [poisson_rand_var(lam) for _ in range(sample_num)]

# plot the histogram of the random variables in the Poisson distribution
plt.figure(figsize=(8, 6))
plt.hist(poisson_rand_vars, bins=range(max(poisson_rand_vars)+1), density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Randomly Obtained Poisson Distribution')

# plot the theoretical Poisson distribution
x_plot = np.arange(0, max(poisson_rand_vars) + 1)
plt.plot(x_plot, poisson_pdf(x_plot, lam),'o-', color='red', label='Theoretical Poisson Distribution')

# config
plt.title(f'Poisson Distribution with λ = {lam}')
plt.xlabel('k')
plt.ylabel('P_k')
plt.legend()
plt.grid()
plt.show()