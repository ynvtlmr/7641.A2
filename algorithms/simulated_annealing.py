import numpy as np
from time import time


class GeomDecay:
    """
    Schedule for geometrically decaying the simulated
    annealing temperature parameter T according to the formula:

    .. math::

        T(t) = \\max(T_{0} \\times r^{t}, T_{min})

    where:

    * :math:`T_{0}` is the initial temperature (at time t = 0);
    * :math:`r` is the rate of geometric decay; and
    * :math:`T_{min}` is the minimum temperature value.

    Parameters
    ----------
    init_temp: float, default: 1.0
        Initial value of temperature parameter T. Must be greater than 0.
    decay: float, default: 0.99
        Temperature decay parameter, r. Must be between 0 and 1.
    min_temp: float, default: 0.001
        Minimum value of temperature parameter. Must be greater than 0.

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        >>> import mlrose
        >>> schedule = mlrose.GeomDecay(init_temp=10, decay=0.95, min_temp=1)
        >>> schedule.evaluate(5)
        7.73780...
    """

    def __init__(self, init_temp=1.0, decay=0.99, min_temp=0.001):

        self.init_temp = init_temp
        self.decay = decay
        self.min_temp = min_temp

        if self.init_temp <= 0:
            raise Exception("""init_temp must be greater than 0.""")

        if (self.decay <= 0) or (self.decay > 1):
            raise Exception("""decay must be between 0 and 1.""")

        if self.min_temp < 0:
            raise Exception("""min_temp must be greater than 0.""")
        elif self.min_temp > self.init_temp:
            raise Exception("""init_temp must be greater than min_temp.""")

    def evaluate(self, t):
        """Evaluate the temperature parameter at time t.

        Parameters
        ----------
        t: int
            Time at which the temperature paramter T is evaluated.

        Returns
        -------
        temp: float
            Temperature parameter at time t.
        """

        temp = self.init_temp * (self.decay ** t)

        if temp < self.min_temp:
            temp = self.min_temp

        return temp


def simulated_annealing(problem,
                        schedule=GeomDecay(),
                        max_attempts=10,
                        max_iters=np.inf,
                        init_state=None):
    """Use simulated annealing to find the optimum for a given
    optimization problem.

    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()` or
        :code:`TSPOpt()`.
    schedule: schedule object, default: :code:`mlrose.GeomDecay()`
        Schedule used to determine the value of the temperature parameter.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better neighbor at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.
    init_state: array, default: None
        1-D Numpy array containing starting state for algorithm.
        If :code:`None`, then a random state is used.

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
    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
            or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
        and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    if init_state is not None and len(init_state) != problem.get_length():
        raise Exception("""init_state must have same length as problem.""")

    # Initialize problem, time and attempts counter
    if init_state is None:
        problem.reset()
    else:
        problem.set_state(init_state)

    attempts = 0
    iters = 0

    iter_fitness = []
    iter_states = []
    iter_time = []
    start_time = time()

    while (attempts < max_attempts) and (iters < max_iters):
        temp = schedule.evaluate(iters)
        iters += 1

        if temp == 0:
            break

        else:
            # Find random neighbor and evaluate fitness
            next_state = problem.random_neighbor()
            next_fitness = problem.eval_fitness(next_state)

            # Calculate delta E and change prob
            delta_e = next_fitness - problem.get_fitness()
            prob = np.exp(delta_e / temp)

            # iterative performance storage
            iter_fitness.append(problem.get_fitness())
            iter_states.append(next_state)
            iter_time.append(time() - start_time)

            # If best neighbor is an improvement or random value is less
            # than prob, move to that state and reset attempts counter
            if (delta_e > 0) or (np.random.uniform() < prob):
                problem.set_state(next_state)
                attempts = 0

            else:
                attempts += 1

    best_fitness = problem.get_maximize() * problem.get_fitness()
    best_state = problem.get_state()

    return best_state, best_fitness, iter_fitness, np.array(iter_states), iter_time
