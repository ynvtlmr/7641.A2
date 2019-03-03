import os
import csv
from time import time
import mlrose
import numpy as np
# import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

filename = "../data/contraceptive.csv"
data = np.loadtxt(filename, delimiter=",", skiprows=1)

if not os.path.exists('../plots/neural_network'):
    os.makedirs('../plots/neural_network')

f = open('../plots/neural_network/values.csv', 'a')
writer = csv.writer(f)

# data = np.genfromtxt(filename, delimiter=',', names=True)

X = data[1:, :-1]
y = data[1:, -1]

"""## Data Normalization"""

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=3)

# Normalize feature data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One hot encode target values
one_hot = OneHotEncoder()
y_train = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
y_test = one_hot.transform(y_test.reshape(-1, 1)).todense()

"""## Neural Networks"""
# Initialize neural network object and fit object

algorithms = [
    'random_hill_climb',
    'simulated_annealing',
    'genetic_alg',
    'gradient_descent'
]

max_iter = 1000
max_attempts = 50

full_run = []
early_quit = []

# values = []
for a in algorithms:
    for i in range(7):
        for early_stopping in [True, False]:
            print('----', a, '----', max_iter, '----', max_attempts, '----')

            start_train = time()
            nn_model1 = mlrose.NeuralNetwork(hidden_nodes=[16],
                                             activation='relu',
                                             algorithm=a,
                                             max_iters=max_iter,
                                             bias=True,
                                             is_classifier=True,
                                             learning_rate=0.01,
                                             early_stopping=early_stopping,
                                             clip_max=5,
                                             pop_size=4,
                                             max_attempts=max_attempts)
            nn_model1.fit(X_train, y_train)
            start_predict = time()
            train_time = start_predict - start_train

            # Predict labels for train set and assess accuracy
            y_train_pred = nn_model1.predict(X_train)
            y_train_accuracy = accuracy_score(y_train, y_train_pred)

            # print(y_train_accuracy)

            # Predict labels for test set and assess accuracy
            y_test_pred = nn_model1.predict(X_test)
            y_test_accuracy = accuracy_score(y_test, y_test_pred)

            # print(y_test_accuracy)
            predict_time = time() - start_predict
            # values.append[a, i, t, y_train_accuracy, y_test_accuracy]

            full_run.append([a, i,
                             max_iter, max_attempts, early_stopping,
                             train_time, predict_time,
                             y_train_accuracy, y_test_accuracy])
            print('full_run', full_run[-1])
            writer.writerow(full_run[-1])

    avg_full_run = np.array(full_run)[:, -4:]
    avg_early_quit = np.array(early_quit)[:, -4:]
    avg_full_run = list(np.mean(avg_full_run.astype(float), axis=0))
    avg_early_quit = list(np.mean(avg_early_quit.astype(float), axis=0))

    writer.writerow(avg_full_run)
    writer.writerow(avg_early_quit)
    print()
f.close()
