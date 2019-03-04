from algorithms.genetic_algorithm import genetic_alg
import mlrose
import random
import numpy as np
import matplotlib.pyplot as plt

# Globals
schedule = mlrose.ExpDecay()
fig_dir = '../plots/'

num_items = 10

"""## Define a Fitness Function Object"""
# weights = [10, 5, 2, 8, 15]
# values = [1, 2, 3, 4, 5]
# max_weight_pct = 0.6

# weights = list(random.sample(range(1, 50), num_items))
# values = list(random.sample(range(1, 20), num_items))

weights = [24, 41, 26, 21, 12, 43, 14, 10, 4, 20]
values = [5, 16, 7, 2, 11, 12, 19, 17, 10, 18]

max_weight_pct = 0.6

fitness = mlrose.Knapsack(weights, values, max_weight_pct)
problem = mlrose.DiscreteOpt(length=num_items,
                             fitness_fn=fitness,
                             maximize=True,
                             max_val=3)


# state = np.array([1, 0, 2, 1, 0])
# print(fitness.evaluate(state))


def solver(problem, population=4, max_attempts=1000, max_iters=1000):
    # genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=10, max_iters=np.inf)
    GA_best_state, GA_best_fitness, GA_iter_fitness, GA_iter_states, GA_iter_time = \
        genetic_alg(problem,
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
pops = [4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80]
iters = 100
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
plt.title("Genetic Algorithm on Knapsack (iter={})".format(iters))

ax.plot(datasets[:, 0], datasets[:, 1])
plt.savefig('../plots/ind_puzzles/GA_Knapsack.png')
plt.show()
plt.close()
