import numpy as np
from time import time


def genetic_alg(problem,
                pop_size=200,
                mutation_prob=0.1,
                max_attempts=10,
                max_iters=np.inf):
    """Use a standard genetic algorithm to find the optimum for a given
    optimization problem.

    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()` or
        :code:`TSPOpt()`.
    pop_size: int, default: 200
        Size of population to be used in genetic algorithm.
    mutation_prob: float, default: 0.1
        Probability of a mutation at each element of the state vector
        during reproduction, expressed as a value between 0 and 1.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better state at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.

    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.

    References
    ----------
    Russell, S. and P. Norvig (2010). *Artificial Intelligence: A Modern
    Approach*, 3rd edition. Prentice Hall, New Jersey, USA.
    """
    if pop_size < 0:
        raise Exception("""pop_size must be a positive integer.""")
    elif not isinstance(pop_size, int):
        if pop_size.is_integer():
            pop_size = int(pop_size)
        else:
            raise Exception("""pop_size must be a positive integer.""")

    if (mutation_prob < 0) or (mutation_prob > 1):
        raise Exception("""mutation_prob must be between 0 and 1.""")

    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
            or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
        and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    # Initialize problem, population and attempts counter
    problem.reset()
    problem.random_pop(pop_size)
    attempts = 0
    iters = 0

    iter_fitness = []
    iter_states = []
    iter_time = []
    start_time = time()

    while (attempts < max_attempts) and (iters < max_iters):
        iters += 1

        # Calculate breeding probabilities
        problem.eval_mate_probs()

        # Create next generation of population
        next_gen = []

        for _ in range(pop_size):
            # Select parents
            selected = np.random.choice(pop_size, size=2,
                                        p=problem.get_mate_probs())
            parent_1 = problem.get_population()[selected[0]]
            parent_2 = problem.get_population()[selected[1]]

            # Create offspring
            child = problem.reproduce(parent_1, parent_2, mutation_prob)
            next_gen.append(child)

        next_gen = np.array(next_gen)
        problem.set_population(next_gen)

        next_state = problem.best_child()
        next_fitness = problem.eval_fitness(next_state)

        # iterative performance storage
        iter_fitness.append(problem.get_fitness())
        iter_states.append(next_state)
        iter_time.append(time() - start_time)

        # If best child is an improvement,
        # move to that state and reset attempts counter
        if next_fitness > problem.get_fitness():
            problem.set_state(next_state)
            attempts = 0

        else:
            attempts += 1

    best_fitness = problem.get_maximize() * problem.get_fitness()
    best_state = problem.get_state()

    return best_state, best_fitness, iter_fitness, np.array(iter_states), iter_time
