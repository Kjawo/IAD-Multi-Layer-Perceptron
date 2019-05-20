import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import NeuralNetwork as nn
import prepData


def digits():
    train_data = pd.read_csv('digits-data/digits-train.csv', header=None)
    test_data = pd.read_csv('digits-data/digits-test.csv', header=None)

    train_target_matrix = np.zeros((train_data.shape[0], 10))
    test_target_matrix = np.zeros((test_data.shape[0], 10))

    for row in range(train_target_matrix.shape[0]):
        train_target_matrix[row][train_data[0][row]] = 1

    for row in range(test_target_matrix.shape[0]):
        test_target_matrix[row][test_data[0][row]] = 1

    train_data = train_data.drop(columns=[0])
    test_data = test_data.drop(columns=[0])
    train_data *= 0.99 / 255
    test_data *= 0.99 / 255

    train_input_matrix = np.asmatrix(train_data.as_matrix())
    test_input_matrix = np.asmatrix(test_data.as_matrix())

    return train_input_matrix, train_target_matrix, test_input_matrix, test_target_matrix


start = time.time()

topology = [50, 10]
bias = True
_lambda = 0.3
_momentum = 0.6
sciezka = 'minist-digits-test'

# train_input_matrix, train_target_matrix, test_input_matrix, test_target_matrix = prepData.iris()

train_input_matrix, train_target_matrix, test_input_matrix, test_target_matrix = digits()

nn.learn(1, topology, train_input_matrix, train_target_matrix, _lambda, _momentum, bias, 1, 0.001, sciezka, False)
nn.test(test_input_matrix, test_target_matrix, topology, sciezka)
# nn.test(train_input_matrix, train_target_matrix, topology, sciezka)

end = time.time()
print('\nExec time: ' + "%0.2f" % (end - start) + 's')