import numpy as np
import matplotlib.pyplot as plt

sum = 0
x_plot = []
ranges = np.arange(1, 1001) # range of n
for n in ranges:
    if np.random.uniform(0, 1) < 0.4: # if win, sum plus 1
        sum += 1
    x_plot.append(sum/n)

# plot the convergence of win rate
plt.figure(figsize=(8, 6))
plt.plot(ranges, x_plot, color='blue', label='win rate')

# config
plt.title(f'Win Rate Convergence (n={ranges[-1]})')
plt.xlabel('n')
plt.ylabel('win/n')
plt.grid()
plt.show()