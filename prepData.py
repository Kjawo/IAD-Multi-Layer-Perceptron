import numpy as np
import pandas as pd
from sklearn import preprocessing


def encoder():
    encoder = np.asmatrix([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
    return encoder, encoder, encoder, encoder


def xor():
    xor_input = np.asmatrix([[1, 0],
                             [0, 1],
                             [0, 0],
                             [1, 1]])

    xor_output = np.asmatrix([[1],
                              [1],
                              [0],
                              [0]])

    return xor_input, xor_output, xor_input, xor_output


def iris():
    file = pd.read_csv('iris.data', header=None)
    output = file[4]

    train_data = file[file.index % 3 != 0]
    test_data = file[file.index % 3 == 0]

    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    train_target_matrix = np.zeros((train_data.shape[0], 3))
    test_target_matrix = np.zeros((test_data.shape[0], 3))

    train_output = train_data[4]
    test_output = test_data[4]

    for row in range(train_output.size):
        train_target_matrix[row][train_output[row]] = 1

    for row in range(test_output.size):
        test_target_matrix[row][test_output[row]] = 1

    train_data = train_data.drop(columns=[4])
    test_data = test_data.drop(columns=[4])

    file = file.drop(columns=[4])

    train_data = pd.DataFrame(preprocessing.scale(train_data))
    test_data = pd.DataFrame(preprocessing.scale(test_data))

    train_input_matrix = np.asmatrix(train_data.as_matrix())
    test_input_matrix = np.asmatrix(test_data.as_matrix())

    return train_input_matrix, train_target_matrix, test_input_matrix, test_target_matrix


def seeds():
    file = pd.read_csv('seeds_dataset.csv', header=None)
    output = file[4]

    train_data = file[file.index % 3 != 0]
    test_data = file[file.index % 3 == 0]

    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    train_target_matrix = np.zeros((train_data.shape[0], 3))
    test_target_matrix = np.zeros((test_data.shape[0], 3))

    train_output = train_data[7]
    test_output = test_data[7]

    for row in range(train_output.size):
        train_target_matrix[row][train_output[row] - 1] = 1

    for row in range(test_output.size):
        test_target_matrix[row][test_output[row] - 1] = 1

    train_data = train_data.drop(columns=[7])
    test_data = test_data.drop(columns=[7])

    file = file.drop(columns=[7])

    train_data = pd.DataFrame(preprocessing.scale(train_data))
    test_data = pd.DataFrame(preprocessing.scale(test_data))

    train_input_matrix = np.asmatrix(train_data.as_matrix())
    test_input_matrix = np.asmatrix(test_data.as_matrix())

    return train_input_matrix, train_target_matrix, test_input_matrix, test_target_matrix
