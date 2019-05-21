import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# import importlib
# importlib.reload(module)


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


def showDigit(input_matrix, correct, guessed):
    label = 'Correct: ' + str(correct) + ' Guessed: ' + str(guessed)
    img = input_matrix.reshape((28, 28))
    plt.imshow(img, cmap="Greys")
    plt.title(label)
    plt.show()