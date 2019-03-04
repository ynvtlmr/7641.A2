import mlrose
from puzzles.puzzle_solver import solver, avg_solver
import random

# Globals
num_items = 10
max_iters = 100
max_attempts = 5

"""## Define a Fitness Function Object"""
# weights = [10, 5, 2, 8, 15]
# values = [1, 2, 3, 4, 5]
# max_weight_pct = 0.6

# weights = list(random.sample(range(1, 50), num_items))
# values = list(random.sample(range(1, 20), num_items))

weights = [24, 41, 26, 21, 12, 43, 14, 10, 4, 20]
values = [5, 16, 7, 2, 11, 12, 19, 17, 10, 18]

max_weight_pct = 0.6

print(weights)
print(values)

fitness = mlrose.Knapsack(weights, values, max_weight_pct)
problem = mlrose.DiscreteOpt(length=num_items,
                             fitness_fn=fitness,
                             maximize=True,
                             max_val=3)

# solver(problem=problem, max_attempts=max_attempts, max_iters=max_iters)
avg_solver(problem=problem)

# state = np.array([1, 0, 2, 1, 0])
# print(fitness.evaluate(state))
