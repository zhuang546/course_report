import numpy as np
import matplotlib.pyplot as plt

# config
sample_num = 1000000 # number of random samples
bins_num = 30 # number of bins in histograms

# define the analytical PDF and CDF
def f(x):
    x = np.array(x)
    result = np.zeros_like(x, dtype=float)
    result[np.abs(x)<=141] = 1/(np.pi * np.sqrt(141**2 - x[np.abs(x)<=141]**2))
    return result
def F(x):
    x = np.array(x)
    result = np.zeros_like(x, dtype=float)
    result[np.abs(x)<=141] = 0.5 + (1 / np.pi) * np.arcsin(x[np.abs(x)<=141] / 141)
    result[x > 141] = 1.0
    return result

#--------------
# plot the PDF
#--------------
plt.figure(figsize=(8, 6))
# generate random samples and calculate x
theta = np.random.uniform(-np.pi/2, np.pi/2, sample_num)
x = 141 * np.sin(theta)
# plot histogram (numrical results for PDF)
plt.hist(x, bins = bins_num, range = [-141,141], density=True, edgecolor='black', color='skyblue', label='numrical results')
# plot PDF curve(analytical results for PDF)
x_plot = np.linspace(-141, 141, 1000)
plt.plot(x_plot, f(x_plot), 'r', label='analytical results')
# config
plt.xticks([-141, 0, 141])
plt.xlim(-141*1.1, 141*1.1)
plt.ylim(0, 0.01)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("PDF")
plt.legend()

#--------------
# plot the CDF
#--------------
plt.figure(figsize=(8, 6))
# plot cumulative histogram (numrical results for CDF)
plt.hist(x, bins=bins_num, range = [-141,141], cumulative=True, density=True, edgecolor='black', color='skyblue', label='numrical results')
# plot CDF curve (analytical results for CDF)
x_plot = np.linspace(-141, 141*1.1, 1000)
plt.plot(x_plot, F(x_plot), 'r', label='analytical results')
# config
plt.xticks([-141, 0, 141])
plt.xlim(-141*1.1, 141*1.1)
plt.ylim(0, 1.1)
plt.xlabel("x")
plt.ylabel("F(x)")
plt.title("CDF")
plt.legend()

# show the plot
plt.show()