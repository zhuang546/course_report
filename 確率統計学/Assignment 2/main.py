import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

sample_num = 1000000
bins_num = 10
n = 5

y= np.array([0.0]*sample_num)
for i in range(n):
    # Generate random x and add it to y
    y += np.random.uniform(-1.0, 1.0, sample_num)
    # plot histogram of y
    plt.figure(figsize=(8, 6))
    plt.hist(y, bins = bins_num*(i+1), density=True, edgecolor='black', color='skyblue', label=f'Yn distribution')
    # plot normal distribution
    x_plot = np.linspace(-5, 5, 1000)
    plt.plot(x_plot, norm.pdf(x_plot, 0, np.sqrt((i+1)/3)), 'r', label='normal distribution')
    # plot config
    plt.title(f'n = {i+1}')
    plt.xlim(-5.0, 5.0)
    plt.ylim(0, 0.55)
    plt.legend()

plt.show()