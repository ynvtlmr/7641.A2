
import networkx as nx
from algorithms.simulated_annealing import simulated_annealing
import numpy as np
import matplotlib.pyplot as plt

import mlrose
from puzzles.puzzle_solver import solver, avg_solver

# Globals
max_iters = 100
max_attempts = 5

# Create list of city coordinates
coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]
# Initialize fitness function object using coords_list
fitness = mlrose.TravellingSales(coords=coords_list)

# Define optimization problem object
problem = mlrose.TSPOpt(length=8, fitness_fn=fitness, maximize=False)


def solver(problem, schedule=None, max_attempts=1, max_iters=1):
    # genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=10, max_iters=np.inf)
    GA_best_state, GA_best_fitness, GA_iter_fitness, GA_iter_states, GA_iter_time = \
        simulated_annealing(problem,
                            max_attempts=max_attempts,
                            max_iters=max_iters,
                            schedule=schedule,
                            )
    # print(problem.fitness_fn.__class__.__name__, "\t genetic_alg-4 \t\t\t",
    #       problem.length,
    #       len(GA_iter_fitness),
    #       GA_best_fitness,
    #       GA_best_state)

    return GA_best_fitness


datasets = []

avg = 100
iter = 100

for i in np.linspace(0.001, 0.03, 60):
    avg_data = []
    schedule = mlrose.ExpDecay(exp_const=i)
    for _ in range(avg):
        avg_data.append(solver(problem, schedule, max_attempts=iter, max_iters=iter))
    avg_data = np.mean(avg_data)
    datasets.append([i, avg_data])
datasets = np.array(datasets)

print(datasets)

fig = plt.figure()  # Create matplotlib figure
ax = fig.add_subplot(111)  # Create matplotlib axes
ax.set_xlabel("Schedule Exponential Decay")
ax.set_ylabel('Fitness (mean, n={})'.format(avg))
plt.title("Simulated Annealing on TSP (iter={})".format(iter))

ax.plot(datasets[:, 0], datasets[:, 1])
plt.savefig('../plots/ind_puzzles/SA_TSP.png')
plt.show()
plt.close()
