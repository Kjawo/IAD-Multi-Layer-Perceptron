import numpy as np
import pandas as pd
from sklearn import preprocessing

def encoder():
    encoder = np.asmatrix([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
    return encoder, encoder


def xor():
    xor_input = np.asmatrix([[1, 0],
                             [0, 1],
                             [0, 0],
                             [1, 1]])

    xor_output = np.asmatrix([[1],
                              [1],
                              [0],
                              [0]])

    return  xor_input, xor_output


def iris():
    file = pd.read_csv('iris.data', header=None)
    output = file[4]
    target_matrix = np.zeros((file.shape[0], 3))

    for row in range(output.size):
        target_matrix[row][output[row]] = 1

    file = file.drop(columns=[4])

    file = pd.DataFrame(preprocessing.scale(file))

    input_matrix = np.asmatrix(file.as_matrix())
    return input_matrix, target_matrix
