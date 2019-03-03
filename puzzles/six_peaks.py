import mlrose
from puzzles.puzzle_solver import solver, avg_solver

# Globals
max_iters = 100
max_attempts = 5

"""## Define a Fitness Function Object"""
fitness = mlrose.SixPeaks(t_pct=0.3)
problem = mlrose.DiscreteOpt(length=10,
                             fitness_fn=fitness,
                             maximize=True,
                             max_val=2)

# solver(problem=problem, max_attempts=max_attempts, max_iters=max_iters)
avg_solver(problem=problem)
