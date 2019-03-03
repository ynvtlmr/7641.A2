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

# solver(problem=problem, max_attempts=max_attempts, max_iters=max_iters)
avg_solver(problem=problem)