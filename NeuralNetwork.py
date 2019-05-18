import numpy as np
import matplotlib.pyplot as plt
import pickle


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    z, y = x.shape
    d = np.asmatrix(np.zeros((z, y)))
    for i in range(z):
        for j in range(y):
            d[i, j] = x[i, j] * (1.0 - x[i, j])
    return d


class NeuronLayer:
    def __init__(self, neurons_count, inputs_per_neuron, is_Bias):
        self.weights = (2 * np.asmatrix(np.random.random((neurons_count, inputs_per_neuron))) - 1) / 2
        # self.bias = (2 * np.asmatrix(np.random.random((ileNeuronow, 1))) - 1) / 2
        self.bias = np.asmatrix(np.zeros((neurons_count, 1)))
        # if is_Bias:
        #     for i in range(0, self.bias.size):
        #         self.bias[i][0] = 1.0
        if is_Bias:
            self.bias_value = 1
        else:
            self.bias_value = 0

        self.input = np.asmatrix([])
        # self.bias = (2 * np.asmatrix(np.random.random((ile_neuronow, 1))).astype(np.float32) - 1) / 2
        self.output = np.asmatrix([])
        self.error = np.matrix([])
        self.v = np.asmatrix(np.zeros((neurons_count, inputs_per_neuron)))
        self.vb = np.asmatrix(np.zeros((neurons_count, 1)))


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
        self.layers[0].v = _lambda * np.multiply(
            sigmoid_derivative(self.layers[0].output),
            self.layers[0].error) * input_matrix.T + _momentum * self.layers[0].v
        self.layers[0].vb = _lambda * np.multiply(
            sigmoid_derivative(self.layers[0].output),
            self.layers[0].error) + _momentum * self.layers[0].vb
        self.layers[0].weights = self.layers[0].weights + self.layers[0].v
        self.layers[0].bias = self.layers[0].bias + self.layers[0].vb
        for i in range(1, len(self.layers)):
            self.layers[i].v = _lambda * np.multiply(
                sigmoid_derivative(self.layers[i].output),
                self.layers[i].error) * self.layers[i - 1].output.T + _momentum * self.layers[i].v
            self.layers[i].vb = _lambda * np.multiply(
                sigmoid_derivative(self.layers[i].output),
                self.layers[i].error) + _momentum * self.layers[i].vb
            self.layers[i].weights = self.layers[i].weights + self.layers[i].v
            self.layers[i].bias = self.layers[i].bias + self.layers[i].vb

    # def get_result(self, result_matix):


# nl = NeuronLayer(2, 3, False)
# print(nl.weights)


def learn(_epoki, _topology, _input_matrix, _target_matrix, _lambda, _momentum, _bias, plot_step, _desired_cost,
          _sciezka, continue_learing):
    df_height, df_width = _input_matrix.shape

    if continue_learing:
        network = pickle.load(open(_sciezka, 'rb'))
    else:
        network = NeuralNetwork(_topology, _bias, df_width)

    ax = list()
    ay = list()
    fig = plt.figure()

    costs = list()

    iterate_list = list(range(df_height))
    for i in range(_epoki):  # epoki
        cost = 0

        for x in iterate_list:
            network.propagate_back(_input_matrix[x].T, _target_matrix[x].reshape(_topology[-1], 1), _lambda, _momentum)
            if (i % plot_step) == 0:
                for q in range(_target_matrix[0].size):
                    cost += (network.layers[-1].error[q, 0] * network.layers[-1].error[q, 0])

        np.random.shuffle(iterate_list)
        if (i % plot_step) == 0:
            ax.append(i)
            ay.append(float(cost))
            print(cost)
            costs.append(cost)
            if cost <= _desired_cost:
                print('Osiągnięto zadany błąd\nIteracja: ', i)
                break

    title = 'Lambda = ' + str(_lambda) + '    Momentum = ' + str(_momentum)

    plt.title(title)
    plt.xlabel('Iteracja')
    plt.ylabel('Błąd')

    plt.plot(ax[0:500], ay[0:500])
    plt.show()

    plt.title(title)
    plt.xlabel('Iteracja')
    plt.ylabel('Błąd')

    plt.plot(ax, ay)
    plt.show()

    pickle.dump(network, open(_sciezka, 'wb'))
    pickle.dump(costs, open(_sciezka + "-costs", 'wb'))


def test(input_matrix, target_matrix, _topology, _sciezka):
    df_height, df_width = input_matrix.shape
    network = pickle.load(open(_sciezka, 'rb'))

    costs = list()
    layers = list()

    avg_cost = 0
    cost = 0
    mistake_count = 0

    np.set_printoptions(suppress=True)
    for i in range(df_height):
        cost = 0
        network.errors(input_matrix[i].T, target_matrix[i].reshape(_topology[-1], 1))
        print("\n", target_matrix[i], ": ")
        print(network.layers[-1].output)
        for q in range(target_matrix[0].size):
            cost += (network.layers[-1].error[q, 0] * network.layers[-1].error[q, 0])
        if (np.where(target_matrix[i] == np.amax(target_matrix[i]))[0]) != \
                (np.where(network.layers[-1].output == np.amax(network.layers[-1].output))[0]):
            mistake_count += 1
        avg_cost += cost
        costs.append(cost)
    avg_cost /= df_height
    print('\nAverage cost: ' + "%0.4f" % avg_cost)
    print('Mistakes count: ' + str(mistake_count))
    print('Sample count: ' + str(df_height))
    print('Correctly guessed: ' + "%0.2f" % ((df_height - mistake_count) / df_height * 100) + "%")

    weights = list()
    for i in reversed(range(len(_topology))):
        weights.append(network.layers[i].weights)

    print('wagi')
    print(network.layers[0].weights)
    print('bias')
    print(network.layers[0].bias)
    print('in')
    print(network.layers[0].input)
    print('out')
    print(network.layers[0].output)
    print('err')
    print(network.layers[0].error)
    print('vb')
    print(network.layers[0].vb)
    print('v')
    print(network.layers[0].v)

    test_data = {'Input matrix': input_matrix,
                 'Weights': weights,
                 'Target matrix': target_matrix,
                 'Cost on every': costs,
                 'Avg cost': avg_cost,
                 'Network:': network
                 }
    pickle.dump(costs, open(_sciezka + "-test-data", 'wb'))
