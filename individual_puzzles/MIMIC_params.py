from algorithms.mimic import mimic
import mlrose
import random
import numpy as np
import matplotlib.pyplot as plt
import mlrose
from puzzles.puzzle_solver import solver, avg_solver
import networkx as nx

# Globals
G = nx.petersen_graph()
nodes = len(G.nodes)
print(nodes)
print(G.edges)

fitness = mlrose.MaxKColor(edges=G.edges)
problem = mlrose.DiscreteOpt(length=nodes,
                             fitness_fn=fitness,
                             maximize=False,
                             max_val=3)

# solver(problem=problem, max_attempts=max_attempts, max_iters=max_iters)


# state = np.array([1, 0, 2, 1, 0])
# print(fitness.evaluate(state))


def solver(problem, population=4, max_attempts=1, max_iters=1):
    # genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=10, max_iters=np.inf)
    GA_best_state, GA_best_fitness, GA_iter_fitness, GA_iter_states, GA_iter_time = \
        mimic(problem,
              max_attempts=max_attempts,
              max_iters=max_iters,
              pop_size=population)
    # print(problem.fitness_fn.__class__.__name__, "\t genetic_alg-4 \t\t\t",
    #       problem.length,
    #       len(GA_iter_fitness),
    #       GA_best_fitness,
    #       GA_best_state)

    return GA_best_fitness


datasets = []
pops = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 56, 64, 80, 96]  # , 128, 192, 256
iters = 3
avg = 100

for i in pops:
    avg_data = []
    for _ in range(avg):
        avg_data.append(solver(problem, i, max_attempts=iters, max_iters=iters))
    avg_data = np.mean(avg_data)
    datasets.append([i, avg_data])
datasets = np.array(datasets)

print(datasets)

fig = plt.figure()  # Create matplotlib figure
ax = fig.add_subplot(111)  # Create matplotlib axes
ax.set_xlabel("Population")
ax.set_ylabel('Fitness (mean, n={})'.format(avg))
plt.title("MIMIC Algorithm on Max-K Color (iter={})".format(iters))

ax.plot(datasets[:, 0], datasets[:, 1])
plt.savefig('../plots/ind_puzzles/MIMIC_Max-K-Color.png')
plt.show()
plt.close()
