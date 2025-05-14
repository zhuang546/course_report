import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

step_num = 100

def get_normal():
    x1 = np.random.uniform(0, 1)
    x2 = np.random.uniform(0, 1)
    y1 = np.sqrt(-2 * np.log(x1)) * np.cos(2 * np.pi * x2)
    y2 = np.sqrt(-2 * np.log(x1)) * np.sin(2 * np.pi * x2)
    return y1, y2

x = np.zeros(step_num)
y = np.zeros(step_num)
for i in range(1, step_num):
    dx, dy = get_normal()
    x[i] = x[i - 1] + dx
    y[i] = y[i - 1] + dy


fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
point, = ax.plot([], [], 'ro')

ax.set_xlim(np.min(x) - 1, np.max(x) + 1)
ax.set_ylim(np.min(y) - 1, np.max(y) + 1)
ax.set_title('Random Walk Simulation')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid()

def init():
    line.set_data([], [])
    point.set_data([], [])
    return line, point

def update(i):
    line.set_data(x[:i+1], y[:i+1])
    point.set_data([x[i]], [y[i]])
    return line, point

ani = animation.FuncAnimation(fig, update, frames=step_num,
                              init_func=init, blit=True, interval=100)

plt.show()

ani.save("random_walk.gif", writer="pillow")
point.set_visible(False)
fig.savefig("final_frame.png")