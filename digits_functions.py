import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.feature import hog
from PIL import Image
from matplotlib import image

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


def digits_hog():
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
    # train_data *= 0.99 / 255
    # test_data *= 0.99 / 255

    train_list_hog_fd = []
    for digit in np.array(train_data, 'int16'):
        fd = hog(digit.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1),
                 visualize=False)
        train_list_hog_fd.append(fd)
    train_hog_features = np.array(train_list_hog_fd, 'float32')

    test_list_hog_fd = []
    for digit in np.array(test_data, 'int16'):
        fd = hog(digit.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1),
                 visualize=False)
        test_list_hog_fd.append(fd)
    test_hog_features = np.array(test_list_hog_fd, 'float32')

    train_input_matrix = np.asmatrix(train_hog_features)
    test_input_matrix = np.asmatrix(test_hog_features)

    return train_input_matrix, train_target_matrix, test_input_matrix, test_target_matrix


def prep_image(image_path):
    three = Image.open(image_path)
    three.thumbnail((28, 28))
    three = three.convert(mode='L')
    three.save('prepared-' + image_path)

    print(three.size)

    data = image.imread('prepared-' + image_path)

    # summarize shape of the pixel array
    print(data.dtype)
    print(data.shape)
    # display the array of pixels as an image

    return data


def prep_image_hog(image_path):
    digit = prep_image(image_path)
    digit *= 255
    hog_of_digit = hog(digit.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1),
                 visualize=False)

    return digit, hog_of_digit


def showDigit(input_matrix, correct, guessed):
    label = 'Correct: ' + str(correct) + ' Guessed: ' + str(guessed)
    img = input_matrix.reshape((28, 28))
    plt.imshow(img, cmap="Greys")
    plt.title(label)
    plt.show()
