from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import numpy as np

problem = get_problem("welded_beam")
algorithm = NSGA2(pop_size=100)
res = minimize(problem,algorithm,
               termination=('n_gen', 200),
               seed=1,
               verbose=False)

# show the solution that is closest to the target (5, 0.002)
target_f1, target_f2 = 5, 0.002
distances = np.sqrt((res.F[:, 0] - target_f1)**2 + (res.F[:, 1] - target_f2)**2)
closest_idx = np.argmin(distances)
solution = res.X[closest_idx]
print(f"Closest solution: h={solution[0]}, l={solution[1]}, t={solution[2]}, b={solution[3]}")
print(f"Corresponding objectives: f1={res.F[closest_idx, 0]}, f2={res.F[closest_idx, 1]}")

# show the pareto front and the solutions
plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="red", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="black")
plot.add(res.F[closest_idx], s=100, facecolor="blue", edgecolor="none")
plot.show()
