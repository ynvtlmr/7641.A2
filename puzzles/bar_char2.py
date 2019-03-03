import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

s = '../plots/fit_time/maximums.csv'
df = pd.read_csv(s, delimiter=',', skipinitialspace=True)

width = 0.25

puzzles = df.puzzle.unique()

for p in puzzles:
    fig = plt.figure()  # Create matplotlib figure

    ax = fig.add_subplot(111)  # Create matplotlib axes
    # ax2 = ax.twinx()  # Create another axes that shares the same x-axis as ax.
    ax3 = ax.twinx()  # Create another axes that shares the same x-axis as ax.

    # ax3.spines["right"].set_position(("axes", 1.2))

    df.loc[df['puzzle'] == p].iteration.plot(kind='bar', color='b', ax=ax, width=width, position=1)
    # df.loc[df['puzzle'] == p].time.plot(kind='bar', color='r', ax=ax2, width=width, position=0.5)
    df.loc[df['puzzle'] == p].maximum.plot(kind='bar', color='g', ax=ax3, width=width, position=0)

    ax.set_xticklabels(df.loc[df['puzzle'] == p].algorithm)

    ax.set_xlabel("Algorithm")

    ax.set_ylabel('Iteration')
    # ax2.set_ylabel('Time')
    ax3.set_ylabel('Maximum')

    # ax2.set_ylim(0, 1)

    ax.yaxis.label.set_color(color="b")
    # ax2.yaxis.label.set_color(color="r")
    ax3.yaxis.label.set_color(color="g")

    ax.tick_params(axis='y', colors="b")
    # ax2.tick_params(axis='y', colors="r")
    ax3.tick_params(axis='y', colors="g")

    plt.title(p)

    plt.show()
    plt.close()
