import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

s = '../plots/neural_network/values.csv'
df = pd.read_csv(s, delimiter=',', skipinitialspace=True)

fig = plt.figure()  # Create matplotlib figure
ax = fig.add_subplot(111)  # Create matplotlib axes

algorithms = df.algorithm.unique()
for a in algorithms:
    # [algorithm, max_iters, time, y_train_accuracy, y_test_accuracy]
    ax.plot(df.loc[df.algorithm == a].max_iters, df.loc[df.algorithm == a].y_train_accuracy)
    # ax.set_xticklabels(df.loc[df['puzzle'] == p].algorithm)
    #
    # ax.set_xlabel("Algorithm")
    #
    # ax.set_ylabel('Iteration')
    # # ax2.set_ylabel('Time')
    # ax3.set_ylabel('Maximum')
    #
    # # ax2.set_ylim(0, 1)
    #
    # ax.yaxis.label.set_color(color="b")
    # # ax2.yaxis.label.set_color(color="r")
    # ax3.yaxis.label.set_color(color="g")
    #
    # ax.tick_params(axis='y', colors="b")
    # # ax2.tick_params(axis='y', colors="r")
    # ax3.tick_params(axis='y', colors="g")
    #
    # plt.title(p)

plt.show()
plt.close()
