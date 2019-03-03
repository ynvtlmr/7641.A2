import os
import numpy as np
import mlrose
import matplotlib.pyplot as plt
import csv

from algorithms.random_hill_climb import random_hill_climb
from algorithms.simulated_annealing import simulated_annealing
from algorithms.genetic_algorithm import genetic_alg
from algorithms.mimic import mimic

schedule = mlrose.ExpDecay()
fig_dir = '../plots/'


def solver(problem, max_attempts=1000, max_iters=1000):
    # random_hill_climb(problem, max_attempts=10, max_iters=np.inf, restarts=0, init_state=None)
    RHC_best_state, RHC_best_fitness, RHC_iter_fitness, RHC_iter_states, RHC_iter_time = \
        random_hill_climb(problem,
                          max_attempts=max_attempts,
                          max_iters=max_iters)
    print(problem.fitness_fn.__class__.__name__, "\t random_hill_climb \t\t",
          problem.length,
          len(RHC_iter_fitness),
          RHC_best_fitness,
          RHC_best_state)

    # simulated_annealing(problem, schedule=GeomDecay(), max_attempts=10, max_iters=np.inf, init_state=None)
    SA_best_state, SA_best_fitness, SA_iter_fitness, SA_iter_states, SA_iter_time = \
        simulated_annealing(problem,
                            schedule=schedule,
                            max_attempts=max_attempts,
                            max_iters=max_iters)
    print(problem.fitness_fn.__class__.__name__, "\t simulated_annealing \t",
          problem.length,
          len(SA_iter_fitness),
          SA_best_fitness,
          SA_best_state)

    # genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=10, max_iters=np.inf)
    GA4_best_state, GA4_best_fitness, GA4_iter_fitness, GA4_iter_states, GA4_iter_time = \
        genetic_alg(problem,
                    max_attempts=max_attempts,
                    max_iters=max_iters,
                    pop_size=4)
    print(problem.fitness_fn.__class__.__name__, "\t genetic_alg-4 \t\t\t",
          problem.length,
          len(GA4_iter_fitness),
          GA4_best_fitness,
          GA4_best_state)

    # genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=10, max_iters=np.inf)
    # GA16_best_state, GA16_best_fitness, GA16_iter_fitness, GA16_iter_states, GA16_iter_time = \
    #     genetic_alg(problem,
    #                 max_attempts=max_attempts,
    #                 max_iters=max_iters,
    #                 pop_size=4)
    # print(problem.fitness_fn.__class__.__name__, "\t genetic_alg-16 \t\t",
    #       problem.length,
    #       len(GA16_iter_fitness),
    #       GA16_best_fitness,
    #       GA16_best_state)

    # mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=10, max_iters=np.inf)
    MM16_best_state, MM16_best_fitness, MM16_iter_fitness, MM16_iter_states, MM16_iter_time = \
        mimic(problem,
              max_attempts=max_attempts,
              max_iters=max_iters,
              pop_size=16)
    print(problem.fitness_fn.__class__.__name__, "\t mimic-16 \t\t\t\t",
          problem.length,
          len(MM16_iter_fitness),
          MM16_best_fitness,
          MM16_best_state)

    # mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=10, max_iters=np.inf)
    MM128_best_state, MM128_best_fitness, MM128_iter_fitness, MM128_iter_states, MM128_iter_time = \
        mimic(problem,
              max_attempts=max_attempts,
              max_iters=max_iters,
              pop_size=128)
    print(problem.fitness_fn.__class__.__name__, "\t mimic-128 \t\t\t\t",
          problem.length,
          len(MM128_iter_fitness),
          MM128_best_fitness,
          MM128_best_state)

    data = {
        'RHC': [RHC_best_state, RHC_best_fitness, RHC_iter_fitness, RHC_iter_states, RHC_iter_time],
        'SA': [SA_best_state, SA_best_fitness, SA_iter_fitness, SA_iter_states, SA_iter_time],
        'GA4': [GA4_best_state, GA4_best_fitness, GA4_iter_fitness, GA4_iter_states, GA4_iter_time],
        # 'GA16': [GA16_best_state, GA16_best_fitness, GA16_iter_fitness, GA16_iter_states, GA16_iter_time],
        'MM16': [MM16_best_state, MM16_best_fitness, MM16_iter_fitness, MM16_iter_states, MM16_iter_time],
        'MM128': [MM128_best_state, MM128_best_fitness, MM128_iter_fitness, MM128_iter_states,
                  MM128_iter_time]
    }
    return data


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def avg_solver(problem, max_attempts=1000, max_iters=1000, avg_size=13):
    if not os.path.exists('../plots/fit_iter'):
        os.makedirs('../plots/fit_iter')
    if not os.path.exists('../plots/fit_time'):
        os.makedirs('../plots/fit_time')

    f = open('../plots/fit_time/maximums.csv', 'a')

    datasets = []
    for _ in range(avg_size):
        datasets.append(solver(problem, max_attempts, max_iters))

    RHC_avg_fitness = []
    SA_avg_fitness = []
    GA4_avg_fitness = []
    # GA16_avg_fitness = []
    MM16_avg_fitness = []
    MM128_avg_fitness = []

    RHC_avg_time = []
    SA_avg_time = []
    GA4_avg_time = []
    # GA16_avg_time = []
    MM16_avg_time = []
    MM128_avg_time = []

    for d in datasets:
        RHC_avg_fitness.append(d['RHC'][2])
        SA_avg_fitness.append(d['SA'][2])
        GA4_avg_fitness.append(d['GA4'][2])
        # GA16_avg_fitness.append(d['GA16'][2])
        MM16_avg_fitness.append(d['MM16'][2])
        MM128_avg_fitness.append(d['MM128'][2])

        RHC_avg_time.append(d['RHC'][-1])
        SA_avg_time.append(d['SA'][-1])
        GA4_avg_time.append(d['GA4'][-1])
        # GA16_avg_time.append(d['GA16'][-1])
        MM16_avg_time.append(d['MM16'][-1])
        MM128_avg_time.append(d['MM128'][-1])

        # ax.plot(d['RHC'][2], alpha=0.1, color='blue')
        # ax.plot(d['SA'][2], alpha=0.1, color='orange')
        # ax.plot(d['GA'][2], alpha=0.1, color='green')
        # ax.plot(d['MM'][2], alpha=0.1, color='red')

    RHC_avg_fitness = np.array(RHC_avg_fitness).mean(axis=0)
    SA_avg_fitness = np.array(SA_avg_fitness).mean(axis=0)
    GA4_avg_fitness = np.array(GA4_avg_fitness).mean(axis=0)
    # GA16_avg_fitness = np.array(GA16_avg_fitness).mean(axis=0)
    MM16_avg_fitness = np.array(MM16_avg_fitness).mean(axis=0)
    MM128_avg_fitness = np.array(MM128_avg_fitness).mean(axis=0)

    RHC_avg_time = np.array(RHC_avg_time).mean(axis=0)
    SA_avg_time = np.array(SA_avg_time).mean(axis=0)
    GA4_avg_time = np.array(GA4_avg_time).mean(axis=0)
    # GA16_avg_time = np.array(GA16_avg_time).mean(axis=0)
    MM16_avg_time = np.array(MM16_avg_time).mean(axis=0)
    MM128_avg_time = np.array(MM128_avg_time).mean(axis=0)

    fig_4, ax_4 = plt.subplots()

    ax_4.plot(RHC_avg_fitness, label='RHC')  # , color='blue'
    ax_4.plot(SA_avg_fitness, label='SA')  # , color='orange'
    ax_4.plot(GA4_avg_fitness, label='GA')  # , color='green'
    # ax_4.plot(GA16_avg_fitness, label='GA16')  # , color='green'
    ax_4.plot(MM16_avg_fitness, label='MM16')  # , color='red'
    ax_4.plot(MM128_avg_fitness, label='MM128')  # , color='magenta'

    ax_4.set_xlabel("Iterations")
    ax_4.set_ylabel("Fitness (mean, n={})".format(avg_size))
    ax_4.set_title(problem.fitness_fn.__class__.__name__)

    ax_4.legend(loc='lower right')
    # plt.show()
    plt.savefig(os.path.join(fig_dir,
                             'fit_iter',
                             "{}-{}-{}".format(problem.fitness_fn.__class__.__name__, max_iters, avg_size)))
    plt.close()


    fig_1, ax_1 = plt.subplots()

    ax_1.semilogx(RHC_avg_fitness, label='RHC')  # , color='blue'
    ax_1.semilogx(SA_avg_fitness, label='SA')  # , color='orange'
    ax_1.semilogx(GA4_avg_fitness, label='GA')  # , color='green'
    # ax_1.semilogx(GA16_avg_fitness, label='GA16')  # , color='green'
    ax_1.semilogx(MM16_avg_fitness, label='MM16')  # , color='red'
    ax_1.semilogx(MM128_avg_fitness, label='MM128')  # , color='magenta'

    ax_1.set_xlabel("Iterations")
    ax_1.set_ylabel("Fitness (mean, n={})".format(avg_size))
    ax_1.set_title(problem.fitness_fn.__class__.__name__)

    ax_1.legend(loc='lower right')
    # plt.show()
    plt.savefig(os.path.join(fig_dir,
                             'fit_iter',
                             "{}-{}-{}-log".format(problem.fitness_fn.__class__.__name__, max_iters, avg_size)))
    plt.close()

    # fig_2, ax_2 = plt.subplots()
    #
    # ax_2.plot(RHC_avg_time, RHC_avg_fitness, l abel='RHC')  # , color='blue'
    # ax_2.plot(SA_avg_time, SA_avg_fitness, label='SA')  # , color='orange'
    # ax_2.plot(GA_avg_time, GA_avg_fitness, label='GA')  # , color='green'
    # ax_2.plot(MM16_avg_time, MM16_avg_fitness, label='MM16')  # , color='red'
    # ax_2.plot(MM128_avg_time, MM128_avg_fitness, label='MM128')  # , color='magenta'
    #
    # ax_2.set_xlabel("Time")
    # ax_2.set_ylabel("Fitness (mean, n={})".format(avg_size))
    # ax_2.set_title(problem.fitness_fn.__class__.__name__)
    #
    # ax_2.legend(loc='lower right')
    # # plt.show()
    #
    # ax_2.set_xlim(left=0.0, right=MM128_avg_time[3])
    # plt.savefig(os.path.join(fig_dir,
    #                          'fit_time',
    #                          "{}-{}-{}-MM".format(problem.fitness_fn.__class__.__name__, max_iters, avg_size)))
    #
    # ax_2.set_xlim(left=0.0, right=SA_avg_time[-1] * 2)
    # plt.savefig(os.path.join(fig_dir,
    #                          'fit_time',
    #                          "{}-{}-{}-GA".format(problem.fitness_fn.__class__.__name__, max_iters, avg_size)))
    #
    # plt.close()

    fig_3, ax_3 = plt.subplots()

    ax_3.semilogx(RHC_avg_time, RHC_avg_fitness, label='RHC')  # , color='blue'
    ax_3.semilogx(SA_avg_time, SA_avg_fitness, label='SA')  # , color='orange'
    ax_3.semilogx(GA4_avg_time, GA4_avg_fitness, label='GA')  # , color='green'
    # ax_3.semilogx(GA16_avg_time, GA16_avg_fitness, label='GA16')  # , color='green'
    ax_3.semilogx(MM16_avg_time, MM16_avg_fitness, label='MM16')  # , color='red'
    ax_3.semilogx(MM128_avg_time, MM128_avg_fitness, label='MM128')  # , color='magenta'

    ax_3.set_xlabel("Time")
    ax_3.set_ylabel("Fitness (mean, n={})".format(avg_size))
    ax_3.set_title(problem.fitness_fn.__class__.__name__)

    ax_3.legend(loc='lower right')
    # plt.show()

    ax_3.set_xlim(left=0.0, right=MM128_avg_time[len(MM128_avg_time) // 100])
    plt.savefig(os.path.join(fig_dir,
                             'fit_time',
                             "{}-{}-{}-MM-log".format(problem.fitness_fn.__class__.__name__, max_iters,
                                                      avg_size)))

    # ax_3.set_xlim(left=0.0, right=SA_avg_time[-1] * 2)
    # plt.savefig(os.path.join(fig_dir,
    #                          'fit_time',
    #                          "{}-{}-{}-GA-log".format(problem.fitness_fn.__class__.__name__, max_iters, avg_size)))

    plt.close()

    ## Get time value of maximums
    print('maximums')
    print('RHC', np.argmax(RHC_avg_fitness), RHC_avg_time[int(np.argmax(RHC_avg_fitness))])
    print('SA', np.argmax(SA_avg_fitness), SA_avg_time[int(np.argmax(RHC_avg_fitness))])
    print('GA4', np.argmax(GA4_avg_fitness), GA4_avg_time[int(np.argmax(RHC_avg_fitness))])
    # print('GA16', np.argmax(GA16_avg_fitness), GA16_avg_time[int(np.argmax(RHC_avg_fitness))])
    print('MM16', np.argmax(MM16_avg_fitness), MM16_avg_time[int(np.argmax(RHC_avg_fitness))])
    print('MM128', np.argmax(MM128_avg_fitness), MM128_avg_time[int(np.argmax(RHC_avg_fitness))])

    # maximums_labels = [
    #     'MAX_RHC_avg_fitness', 'Time_to_MAX_RHC_avg_fitness',
    #     'MAX_SA_avg_fitness', 'Time_to_MAX_SA_avg_fitness',
    #     'MAX_GA_avg_fitness', 'Time_to_MAX_GA_avg_fitness',
#     #     'MAX_GA16_avg_fitness', 'Time_to_MAX_GA16_avg_fitness',
    #     'MAX_MM16_avg_fitness', 'Time_to_MAX_MM16_avg_fitness',
    #     'MAX_MM128_avg_fitness', 'Time_to_MAX_MM128_avg_fitness',
    # ]
    maximums_labels = [
        'problem', 'algorithm', 'iter_to_max_fitness', 'time_to_max_fitness', 'max_fitness'
    ]
    maximums = [
        [problem.fitness_fn.__class__.__name__, 'RHC',
            np.argmax(RHC_avg_fitness), RHC_avg_time[int(np.argmax(RHC_avg_fitness))],
            np.max(RHC_avg_fitness)],
        [problem.fitness_fn.__class__.__name__, 'SA',
            np.argmax(SA_avg_fitness), SA_avg_time[int(np.argmax(RHC_avg_fitness))],
            np.max(SA_avg_fitness)],
        [problem.fitness_fn.__class__.__name__, 'GA4',
            np.argmax(GA4_avg_fitness), GA4_avg_time[int(np.argmax(RHC_avg_fitness))],
            np.max(GA4_avg_fitness)],
        # [problem.fitness_fn.__class__.__name__, 'GA16',
        #     np.argmax(GA16_avg_fitness), GA16_avg_time[int(np.argmax(RHC_avg_fitness))],
        #    np.max(GA16_avg_fitness)],
        [problem.fitness_fn.__class__.__name__, 'MM16',
            np.argmax(MM16_avg_fitness), MM16_avg_time[int(np.argmax(RHC_avg_fitness))],
            np.max(MM16_avg_fitness)],
        [problem.fitness_fn.__class__.__name__, 'MM128',
            np.argmax(MM128_avg_fitness), MM128_avg_time[int(np.argmax(RHC_avg_fitness))],
            np.max(MM128_avg_fitness)],
    ]

    writer = csv.writer(f)
    # writer.writerow(maximums_labels)
    writer.writerows(maximums)
    # for m in maximums:
    #     writer.writerow(m)
    # np.savetxt(f, np.array(maximums_labels), newline="")
    # np.savetxt(f, np.array(maximums), delimiter=", ", newline=", ")

    f.close()


def difficulty_solver(problem, max_attempts=30, max_iters=1000, difficulties=[8, 16, 32]):
    datasets = []
    for d in difficulties:
        problem.length = d
        datasets.append(solver(problem, max_attempts, max_iters))

    print()


def complexity_time_solver(problem,
                           max_attempts=10,
                           max_iters=10,
                           avg_size=3,
                           complexity=list(range(10, 21, 5))):
    if not os.path.exists('../plots/complex_time'):
        os.makedirs('../plots/complex_time')

    datasets = {
        'RHC': {},
        'SA': {},
        'GA': {},
        'MM16': {},
        'MM128': {},
    }

    for c in complexity:
        datasets['RHC'][c] = []
        datasets['SA'][c] = []
        datasets['GA'][c] = []
        datasets['MM16'][c] = []
        datasets['MM128'][c] = []
        for _ in range(avg_size):
            problem.length = c
            data = solver(problem, max_attempts, max_iters)
            datasets['RHC'][c].append(data['RHC'][-1][-1])
            datasets['SA'][c].append(data['SA'][-1][-1])
            datasets['GA'][c].append(data['GA'][-1][-1])
            datasets['MM16'][c].append(data['MM16'][-1][-1])
            datasets['MM128'][c].append(data['MM128'][-1][-1])

        datasets['RHC'][c] = np.mean(np.array(datasets['RHC'][c]))
        datasets['SA'][c] = np.mean(np.array(datasets['SA'][c]))
        datasets['GA'][c] = np.mean(np.array(datasets['GA'][c]))
        datasets['MM16'][c] = np.mean(np.array(datasets['MM16'][c]))
        datasets['MM128'][c] = np.mean(np.array(datasets['MM128'][c]))
        # datasets.append(solver(problem, max_attempts, max_iters))

    print(datasets)

    # ax.plot(RHC_avg_time, label='RHC')  # , color='blue'
    # ax.plot(SA_avg_time, label='SA')  # , color='orange'
    # ax.plot(GA_avg_time, label='GA')  # , color='green'
    # ax.plot(MM16_avg_time, label='MM16')  # , color='red'
    # ax.plot(MM128_avg_time, label='MM128')  # , color='magenta'
    #
    # ax.set_xlabel("Complexity")
    # ax.set_ylabel("Time (mean, n={})".format(avg_size))
    # ax.set_title(problem.fitness_fn.__class__.__name__)
    #
    # ax.legend(loc='lower right')
    # # plt.show()
    # plt.savefig(os.path.join(fig_dir,
    #                          'complex_time',
    #                          "{}-{}-{}".format(problem.fitness_fn.__class__.__name__, max_iters, avg_size)))

# def fitness_time():
