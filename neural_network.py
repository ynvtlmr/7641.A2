import mlrose

import numpy as np
from sklearn.datasets import load_iris

# Load the Iris dataset
data = load_iris()

filename = "./data/contraceptive.csv"
# filename = "./banana.csv"
data = np.loadtxt(filename, delimiter=",", skiprows=1)

# data = np.genfromtxt(filename, delimiter=',', names=True)

X = data[:, :-1]
y = data[:, -1]

"""## Data Normalization"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=3)

# Normalize feature data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One hot encode target values
one_hot = OneHotEncoder()
y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()

"""## Neural Networks"""

# Initialize neural network object and fit object
np.random.seed(3)


algorithms = [
    'random_hill_climb',
    'simulated_annealing',
    'genetic_alg',
    'gradient_descent'
]

for a in algorithms:
    print('----', a, '----', )

    nn_model1 = mlrose.NeuralNetwork(hidden_nodes=[2],
                                     activation='relu',
                                     algorithm='random_hill_climb',
                                     max_iters=1000,
                                     bias=True,
                                     is_classifier=True,
                                     learning_rate=0.0001,
                                     early_stopping=True,
                                     clip_max=5,
                                     max_attempts=100)
    nn_model1.fit(X_train_scaled, y_train_hot)

    from sklearn.metrics import accuracy_score

    # Predict labels for train set and assess accuracy
    y_train_pred = nn_model1.predict(X_train_scaled)
    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

    print(y_train_accuracy)
    # 0.45

    # Predict labels for test set and assess accuracy
    y_test_pred = nn_model1.predict(X_test_scaled)
    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

    print(y_test_accuracy)
    # 0.533333333333
#
# # Initialize neural network object and fit object
# np.random.seed(3)
# nn_model2 = mlrose.NeuralNetwork(hidden_nodes=[2],
#                                  activation='relu',
#                                  algorithm='gradient_descent',
#                                  max_iters=1000,
#                                  bias=True,
#                                  is_classifier=True,
#                                  learning_rate=0.0001,
#                                  early_stopping=True,
#                                  clip_max=5,
#                                  max_attempts=100)
#
# nn_model2.fit(X_train_scaled, y_train_hot)
#
# # Predict labels for train set and assess accuracy
# y_train_pred = nn_model2.predict(X_train_scaled)
# y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
#
# print(y_train_accuracy)
# # 0.625
#
# # Predict labels for test set and assess accuracy
# y_test_pred = nn_model2.predict(X_test_scaled)
# y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
#
# print(y_test_accuracy)
# # 0.566666666667
#
# """## Linear and Logistic Regression Models
#
# For example, suppose we wished to fit a logistic regression to our Iris data using the randomized hill climbing algorithm and all other parameters set as for the example in the previous section. We could do this by initializing a
# NeuralNetwork() object like so:
# """
#
# lr_nn_model1 = mlrose.NeuralNetwork(hidden_nodes=[],
#                                     activation='sigmoid',
#                                     algorithm='random_hill_climb',
#                                     max_iters=1000,
#                                     bias=True,
#                                     is_classifier=True,
#                                     learning_rate=0.0001,
#                                     early_stopping=True,
#                                     clip_max=5,
#                                     max_attempts=100)
#
# """Actual way to initialize such a network with MLRose"""
#
# # Initialize logistic regression object and fit object
# np.random.seed(3)
# lr_model1 = mlrose.LogisticRegression(algorithm='random_hill_climb',
#                                       max_iters=1000,
#                                       bias=True,
#                                       learning_rate=0.0001,
#                                       early_stopping=True,
#                                       clip_max=5,
#                                       max_attempts=100)
#
# lr_model1.fit(X_train_scaled, y_train_hot)
#
# # Predict labels for train set and assess accuracy
# y_train_pred = lr_model1.predict(X_train_scaled)
# y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
#
# print(y_train_accuracy)
# # 0.191666666667
#
# # Predict labels for test set and assess accuracy
# y_test_pred = lr_model1.predict(X_test_scaled)
# y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
#
# print(y_test_accuracy)
# # 0.0666666666667
#
# # Initialize logistic regression object and fit object
# np.random.seed(3)
# lr_model2 = mlrose.LogisticRegression(algorithm='random_hill_climb',
#                                       max_iters=10000,
#                                       bias=True,
#                                       learning_rate=0.01,
#                                       early_stopping=True,
#                                       clip_max=5,
#                                       max_attempts=100)
#
# lr_model2.fit(X_train_scaled, y_train_hot)
#
# # Predict labels for train set and assess accuracy
# y_train_pred = lr_model2.predict(X_train_scaled)
# y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
#
# print(y_train_accuracy)
# # 0.683333333333
#
# # Predict labels for test set and assess accuracy
# y_test_pred = lr_model2.predict(X_test_scaled)
# y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
#
# print(y_test_accuracy)
# # 0.7
#
# # 0.5521226415094339
# # 0.5537735849056604
