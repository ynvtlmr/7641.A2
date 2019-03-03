import mlrose
from puzzles.puzzle_solver import solver, avg_solver

# Globals
max_iters = 1000
max_attempts = 50

"""## Define a Fitness Function Object"""
fitness = mlrose.Queens()
problem = mlrose.DiscreteOpt(length=8,
                             fitness_fn=fitness,
                             maximize=False,
                             max_val=8
                             )

# solver(problem=problem, max_attempts=max_attempts, max_iters=max_iters)
avg_solver(problem=problem, max_attempts=200, max_iters=200)
