import matplotlib.pyplot as plt
import pandas as pd

s = '../plots/neural_network/avg_NN_13.csv'
df = pd.read_csv(s, delimiter=',', skipinitialspace=True)

width = 0.2

algorithm = df.algorithm.unique()

fig = plt.figure()  # Create matplotlib figure

ax = fig.add_subplot(111)  # Create matplotlib axes

# for a in algorithm:
df.full_train_time.plot(kind='bar', color='b', ax=ax, width=width, position=1, label='full_train')
df.early_train_time.plot(kind='bar', color='g', ax=ax, width=width, position=0, label='early_train')

ax.set_xticklabels(['RHC', 'SA', 'GA', 'SGD'])
#
ax.set_xlabel("Algorithm")
#
ax.set_ylabel('Time (mean, n=13)')
ax.yaxis.label.set_color(color="b")
ax.tick_params(axis='y', colors="b")

ax.set_xlim(-0.5)
# ax.set_ylim(0, 2)

plt.legend(loc='upper right', shadow=False)
plt.title("Neural Network Training Times")

plt.show()
plt.close()

""" ## ~~ ## """


fig = plt.figure()  # Create matplotlib figure

ax = fig.add_subplot(111)  # Create matplotlib axes


df.full_y_train_accuracy.plot(kind='bar', color='b', ax=ax, width=width, position=2.1, label='full_train')
df.full_y_test_accuracy.plot(kind='bar', color='r', ax=ax, width=width, position=1.1, label='full_test')
df.early_y_train_accuracy.plot(kind='bar', color='g', ax=ax, width=width, position=-0.1, label='early_train')
df.early_y_test_accuracy.plot(kind='bar', color='c', ax=ax, width=width, position=-1.1, label='early_test')

ax.set_xticklabels(['RHC', 'SA', 'GA', 'SGD'])
#
ax.set_xlabel("Algorithm")
#
ax.set_ylabel('Accuracy (mean, n=13)')

ax.set_xlim(-0.5)
ax.set_ylim(0.35, 0.5)
ax.yaxis.label.set_color(color="b")

ax.tick_params(axis='y', colors="b")

plt.legend(loc='upper right', shadow=False)
plt.title("Neural Network Train & Test Accuracy")

plt.show()
plt.close()
