import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

sample_num = 100000
bins = 30
n = 1000
lam = 2

# generate random variables by exponential distribution
samples = [np.sum(np.random.exponential(1/lam, n))/n for _ in range(sample_num)]

# plot the histogram
plt.figure(figsize=(8, 6))
plt.hist(samples, bins=bins, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='X^bar by Exponential Distribution')
# plot the theoretical normal distribution
ranges = [1/lam - 3/(lam*np.sqrt(n)), 1/lam + 3/(lam*np.sqrt(n))]
x_plot = np.linspace(ranges[0], ranges[1], 1000)
plt.plot(x_plot, stats.norm.pdf(x_plot, loc=1/lam, scale=1/(lam*np.sqrt(n))), color='red', label='Theoretical Normal Distribution')

# config
plt.title(f'Confirmation of Central Limit Theorem (n={n})')
plt.xlim(ranges)
plt.legend()
plt.show()