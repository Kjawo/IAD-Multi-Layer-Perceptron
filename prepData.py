import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def encoder():
    encoder = np.asmatrix([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
    return encoder, np.asarray(encoder), encoder, np.asarray(encoder)


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

    # file = file.sample(frac=1).reset_index(drop=True)

    file.loc[file[4] == 'Iris-setosa', 4] = 0
    file.loc[file[4] == 'Iris-versicolor', 4] = 1
    file.loc[file[4] == 'Iris-virginica', 4] = 2

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


def knn_iris():
    df = pd.read_csv('iris.data',
                     names=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'])
    X_train, X_test, Y_train, Y_test = train_test_split(df[['SepalLengthCm', 'SepalWidthCm',
                                                            'PetalLengthCm', 'PetalWidthCm']],
                                                        df['Species'], random_state=0)
    return X_train, X_test, Y_train, Y_test


def knn_seeds():
    df = pd.read_csv('seeds_dataset.csv',
                     names=['area', 'perimeter', 'compactness', 'length of kernel', 'width of kernel',
                            'asymmetry coefficient', 'length of kernel groove', 'type'])
    X_train, X_test, Y_train, Y_test = train_test_split(
        df[['area', 'perimeter', 'compactness', 'length of kernel', 'width of kernel',
            'asymmetry coefficient', 'length of kernel groove']],
        df['type'], random_state=0)

    return X_train, X_test, Y_train, Y_test


def sqrt_data(betweenZeroAndOne):
    x = np.array([])
    y = np.array([])

    for i in np.arange(0, 10.0, 10.0/3000):
        x = np.append(x, i)

    y = np.sqrt(x)

    if betweenZeroAndOne:
        # x = np.interp(x, (x.min(), x.max()), (0, +1))
        y = np.interp(y, (y.min(), y.max()), (0, +1))

    x.shape = [x.shape[0], 1]
    y.shape = [y.shape[0], 1]

    test_x = x[::3]
    test_y = y[::3]
    x = np.delete(x, np.arange(0, x.size, 3))
    y = np.delete(y, np.arange(0, y.size, 3))
    train_x = x.reshape(x.shape[0], 1)
    train_y = y.reshape(y.shape[0], 1)

    return train_x, train_y, test_x, test_y


def sin_data(betweenZeroAndOne):
    x = np.array([])
    y = np.array([])

    for i in np.arange(-10.0, 10.0, 20.0/3000):
        x = np.append(x, i)

    y = np.sin(x)

    if betweenZeroAndOne:
        # x = np.interp(x, (x.min(), x.max()), (0, +1))
        y = np.interp(y, (y.min(), y.max()), (0, +1))

    x.shape = [x.shape[0], 1]
    y.shape = [y.shape[0], 1]

    test_x = x[::3]
    test_y = y[::3]
    x = np.delete(x, np.arange(0, x.size, 3))
    y = np.delete(y, np.arange(0, y.size, 3))
    train_x = x
    train_y = y

    return train_x, train_y, test_x, test_y


def sin_cos_data(betweenZeroAndOne):
    #sin(x1 * x2) + cos(3*(x1 - x2)) na przedziale x1: [-3, 3], x2: [-3, 3]
    x = []
    y = np.array([])

    for i in np.arange(-3.0, 3.0, 6.0/100):
        for j in np.arange(-3.0, 3.0, 6.0 / 100):
            x.append([i, j])
    x = np.array(x)

    y = np.sin(x[:, 0] * x[:, 1]) + np.cos(3*(x[:, 0] - x[:, 1]))

    # if betweenZeroAndOne:
    #     # x = np.interp(x, (x.min(), x.max()), (0, +1))
    #     y = np.interp(y, (y.min(), y.max()), (0, +1))
    #
    # x1.shape = [x1.shape[0], 1]
    # y.shape = [y.shape[0], 1]
    #
    # test_x = x[::3]
    # test_y = y[::3]
    # x = np.delete(x, np.arange(0, x.size, 3))
    # y = np.delete(y, np.arange(0, y.size, 3))
    # train_x = x
    # train_y = y

    # return train_x, train_y, test_x, test_y
    return x,y,x,y

