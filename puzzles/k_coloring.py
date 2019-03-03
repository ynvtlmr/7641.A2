import mlrose
from puzzles.puzzle_solver import solver, avg_solver
import networkx as nx

# Globals
max_iters = 100
max_attempts = 5

# nodes = 10
# edges = [(0, 1), (1, 2), (2, 3), (3, 4),
#          (4, 0), (0, 5), (1, 6), (2, 7),
#          (3, 8), (4, 9), (5, 7), (7, 9),
#          (9, 6), (6, 8), (8, 5)]
# print(edges)
# fitness = mlrose.MaxKColor(edges=edges)
# problem = mlrose.DiscreteOpt(length=nodes,
#                              fitness_fn=fitness,
#                              maximize=False,
#                              max_val=3)


# G = nx.random_geometric_graph(50, 0.125)
G = nx.petersen_graph()
# G = nx.tutte_graph()
# G = nx.sedgewick_maze_graph()
# G = nx.tetrahedral_graph()
# G = nx.desargues_graph()
nodes = len(G.nodes)
print(nodes)
print(G.edges)

fitness = mlrose.MaxKColor(edges=G.edges)
problem = mlrose.DiscreteOpt(length=nodes,
                             fitness_fn=fitness,
                             maximize=False,
                             max_val=3)

# solver(problem=problem, max_attempts=max_attempts, max_iters=max_iters)
avg_solver(problem=problem)

# edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
# fitness = mlrose.MaxKColor(edges=edges)
# problem = mlrose.DiscreteOpt(length=5,
#                              fitness_fn=fitness,
#                              maximize=False,
#                              max_val=2)

# Two greedy colorings
# edges = [(0, 5), (0, 6), (0, 7),
#          (1, 4), (1, 6), (1, 7),
#          (2, 4), (2, 5), (2, 7),
#          (3, 4), (3, 5), (3, 6),
#          (4, 1), (4, 2), (4, 3),
#          (5, 0), (5, 2), (5, 3),
#          (6, 0), (6, 1), (6, 3),
#          (7, 0), (7, 1), (7, 2)]
# fitness = mlrose.MaxKColor(edges=edges)
# problem = mlrose.DiscreteOpt(length=8,
#                              fitness_fn=fitness,
#                              maximize=False,
#                              max_val=2)

# Mostly connected graph
# nodes = 32
# edges = []
# for i in range(nodes):
#     edges.append((i, (i + 1) % nodes))
#     num_edges = nodes // 2
#     for j in range(1, num_edges):
#         edges.append((i, (i + (2 * j)) % nodes))
# fitness = mlrose.MaxKColor(edges=edges)
# problem = mlrose.DiscreteOpt(length=nodes,
#                              fitness_fn=fitness,
#                              maximize=False,
#                              max_val=nodes / 2)

# Petersen graph
# nodes = 10
# edges = []
# for i in range(nodes):
#     if i < nodes // 2:
#         edges.append((i, (i + 1) % nodes))
#         edges.append((i, i + nodes // 2))
#     if i + 2 < nodes:
#         edges.append((i, i + 2))
#     if i + 3 < nodes:
#         edges.append((i, i + 3))
# print(edges)
# fitness = mlrose.MaxKColor(edges=edges)
# problem = mlrose.DiscreteOpt(length=nodes,
#                              fitness_fn=fitness,
#                              maximize=False,
#                              max_val=3)
