import pickle

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    z, y = x.shape
    d = np.asmatrix(np.zeros((z, y)))
    for i in range(z):
        for j in range(y):
            d[i, j] = x[i, j] * (1.0 - x[i, j])
            # d[i, j] = sigmoid(x[i, j]) * (1.0 - sigmoid(x[i, j]))
    return d


class NeuronLayer:
    def __init__(self, neurons_count, inputs_per_neuron, is_Bias):
        self.weights = (2 * np.asmatrix(np.random.random((neurons_count, inputs_per_neuron))) - 1) / 2
        # self.bias = (2 * np.asmatrix(np.random.random((ileNeuronow, 1))) - 1) / 2
        self.bias = np.asmatrix(np.zeros((neurons_count, 1)))
        if is_Bias:
            self.bias_value = 1
        else:
            self.bias_value = 0

        self.input = np.asmatrix([])
        # self.bias = (2 * np.asmatrix(np.random.random((ile_neuronow, 1))).astype(np.float32) - 1) / 2
        self.output = np.asmatrix([])
        self.error = np.matrix([])
        self.weight_change = np.asmatrix(np.zeros((neurons_count, inputs_per_neuron)))
        self.weight_change_bias = np.asmatrix(np.zeros((neurons_count, 1)))


# obj.T -> transponowanie macierzy
# diagflat(array) tworzy macierz diagonalną z wyrazami z array na przekątnej


class NeuralNetwork:

    def __init__(self, topology, bias, input_size):
        # self.layers = list(range(np.size(input_matrix, 0)))
        self.layers = list([None] * np.size(topology))
        self.layers[0] = NeuronLayer(topology[0], input_size, bias)
        for i in range(1, len(topology)):
            self.layers[i] = NeuronLayer(topology[i], topology[i - 1], bias)

    def propagate_forward(self, input_matrix):
        self.layers[0].input = self.layers[0].weights * input_matrix + self.layers[0].bias * self.layers[0].bias_value
        self.layers[0].output = sigmoid(self.layers[0].input)
        for i in range(1, len(self.layers)):
            self.layers[i].input = self.layers[i].weights * self.layers[i - 1].output + self.layers[i].bias * \
                                   self.layers[i].bias_value
            self.layers[i].output = sigmoid(self.layers[i].input)

    def errors(self, input_matrix, target_matrix):
        self.propagate_forward(input_matrix)
        self.layers[-1].error = target_matrix - self.layers[-1].output
        for i in reversed(range(len(self.layers) - 1)):
            self.layers[i].error = self.layers[i + 1].weights.T * np.diagflat(
                sigmoid_derivative(self.layers[i + 1].output)) * self.layers[i + 1].error

    def propagate_back(self, input_matrix, target_matrix, _lambda, _momentum):
        self.errors(input_matrix, target_matrix)
        self.layers[0].weight_change = _lambda * np.multiply(
            sigmoid_derivative(self.layers[0].output),
            self.layers[0].error) * input_matrix.T + _momentum * self.layers[0].weight_change
        self.layers[0].weight_change_bias = _lambda * np.multiply(
            sigmoid_derivative(self.layers[0].output),
            self.layers[0].error) + _momentum * self.layers[0].weight_change_bias
        self.layers[0].weights = self.layers[0].weights + self.layers[0].weight_change
        self.layers[0].bias = self.layers[0].bias + self.layers[0].weight_change_bias
        for i in range(1, len(self.layers)):
            self.layers[i].weight_change = _lambda * np.multiply(
                sigmoid_derivative(self.layers[i].output),
                self.layers[i].error) * self.layers[i - 1].output.T + _momentum * self.layers[i].weight_change
            self.layers[i].weight_change_bias = _lambda * np.multiply(
                sigmoid_derivative(self.layers[i].output),
                self.layers[i].error) + _momentum * self.layers[i].weight_change_bias
            self.layers[i].weights = self.layers[i].weights + self.layers[i].weight_change
            self.layers[i].bias = self.layers[i].bias + self.layers[i].weight_change_bias


def learn(_epoki, _topology, _input_matrix, _target_matrix, train_X, train_Y, _lambda, _momentum, _bias, plot_step,
          _desired_cost, _sciezka, continue_learing, plot_acc):
    # df_height, df_width = _input_matrix.shape
    df_height = _input_matrix.shape[0]
    df_width = 1

    if continue_learing:
        network = pickle.load(open(_sciezka, 'rb'))
    else:
        network = NeuralNetwork(_topology, _bias, df_width)

    ax = list()
    ay = list()
    # acc_x = list()
    acc_y = list()
    fig = plt.figure()

    costs = list()

    iterate_list = list(range(df_height))
    for i in range(_epoki):  # epoki
        cost = 0

        for x in tqdm(iterate_list):
            network.propagate_back(_input_matrix[x].T, _target_matrix[x].reshape(_topology[-1], 1), 2 * _lambda,
                                   _momentum)
            if (i % plot_step) == 0:
                for q in range(_target_matrix[0].size):
                    cost += (network.layers[-1].error[q, 0] * network.layers[-1].error[q, 0])
        np.random.shuffle(iterate_list)
        if (i % plot_step) == 0:
            if plot_acc:
                mistake_count = 0
                for j in range(train_X.shape[0]):
                    network.propagate_forward(train_X[j].T)

                    guessed_class = np.where(network.layers[-1].output == np.amax(network.layers[-1].output))[0]
                    correct_class = np.where(train_Y[j] == np.amax(train_Y[j]))[0]

                    if guessed_class.all() != correct_class.all():
                        mistake_count += 1
                acc_y.append(((df_height - mistake_count) / df_height * 100))

            ax.append(i)
            ay.append(float(cost))
            print(cost)
            costs.append(cost)
            if cost <= _desired_cost:
                print('Osiągnięto zadany błąd\nIteracja: ', i)
                break

    title = 'Lambda = ' + str(_lambda) + '    Momentum = ' + str(_momentum) + "width: " + str(df_width)

    if plot_acc:
        plt.title(title)
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')

        plt.plot(ax, acc_y)
        plt.show()

    plt.title(title)
    plt.xlabel('Iteracja')
    plt.ylabel('Koszt')

    plt.plot(ax, ay)
    plt.show()

    pickle.dump(network, open(_sciezka, 'wb'))
    pickle.dump(costs, open(_sciezka + "-costs", 'wb'))


def test(input_matrix, target_matrix, _topology, _sciezka, verbose, is_digit):
    # df_height, df_width = input_matrix.shape
    df_height = input_matrix.shape[0]
    df_width = 1
    network = pickle.load(open(_sciezka, 'rb'))

    costs = list()
    layers = list()

    avg_cost = 0
    cost = 0
    mistake_count = 0

    bledy_i_rodzaju = list([0] * _topology[-1])
    bledy_ii_rodzaju = list([0] * _topology[-1])

    ax = list()
    ay = list()
    fig = plt.figure()

    np.set_printoptions(suppress=True)
    for i in range(df_height):
        cost = 0
        network.errors(input_matrix[i].T, target_matrix[i].reshape(_topology[-1], 1))
        if verbose:
            print("\n", target_matrix[i], ": ")
            print(network.layers[-1].output)
        # encoder
        # print('Output:')
        # print(network.layers[0].output)

        for q in range(target_matrix[0].size):
            cost += (network.layers[-1].error[q, 0] * network.layers[-1].error[q, 0])

        guessed_class = np.where(network.layers[-1].output == np.amax(network.layers[-1].output))[0]
        correct_class = np.where(target_matrix[i] == np.amax(target_matrix[i]))[0]

        if guessed_class.all() != correct_class.all():
            mistake_count += 1
            bledy_i_rodzaju[correct_class[0]] += 1
            bledy_ii_rodzaju[guessed_class[0]] += 1
            if is_digit:
                import digits_functions
                digits_functions.showDigit(input_matrix[i], correct_class, guessed_class)
        avg_cost += cost
        costs.append(cost)
        ax.append(input_matrix[i])
        # ay.append(target_matrix[i])
        ay.append(network.layers[-1].output.__float__())

    print('\nAverage cost: ' + "%0.4f" % avg_cost)
    print('Mistakes count: ' + str(mistake_count))
    print('Bledy I rodzaju: ' + str(bledy_i_rodzaju))
    print('Bledy II rodzaju: ' + str(bledy_ii_rodzaju))
    print('Sample count: ' + str(df_height))
    print('Correctly guessed: ' + "%0.2f" % ((df_height - mistake_count) / df_height * 100) + "%")

    weights = list()
    for i in reversed(range(len(_topology))):
        weights.append(network.layers[i].weights)

    # print('wagi')
    # print(network.layers[0].weights)
    # print('bias')
    # print(network.layers[0].bias)
    # print('in')
    # print(network.layers[0].input)
    # print('out')
    # print(network.layers[0].output)
    # print('err')
    # print(network.layers[0].error)
    # print('weight_change_bias')
    # print(network.layers[0].weight_change_bias)
    # print('v')
    # print(network.layers[0].v)


    plt.xlabel('x')
    plt.ylabel('y')

    plt.plot(ax, ay)
    plt.show()

    test_data = {'Input matrix': input_matrix,
                 'Weights': weights,
                 'Target matrix': target_matrix,
                 'Cost on every': costs,
                 'Avg cost': avg_cost,
                 'Network:': network
                 }
    pickle.dump(test_data, open(_sciezka + "-test-data", 'wb'))
