import numpy as np


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
    def __init__(self, ileNeuronow, ileWejscNaNeuron, isBias):
        self.weights = (2 * np.asmatrix(np.random.random((ileNeuronow, ileWejscNaNeuron))) - 1) / 2
        # self.bias = np.asmatrix(np.zeros((ileNeuronow, 1)))
        # if isBias:
        #     for i in range(0, self.bias.size):
        #        self.bias[i][0] = 1.0
        self.input = np.asmatrix([])
        self.bias = (2 * np.asmatrix(np.random.random((ileNeuronow, 1))).astype(np.float32) - 1) / 2
        self.output = np.asmatrix([])
        self.error = np.matrix([])
        self.v = np.asmatrix(np.zeros((ileNeuronow, ileWejscNaNeuron)))
        self.vb = np.asmatrix(np.zeros((ileNeuronow, 1)))

# obj.T -> transponowanie macierzy
# diagflat(array) tworzy macierz diagonalną z wyrazami z array na przekątnej


class NeuralNetwork:

    def __init__(self, topology, bias, input_matrix):
        # self.layers = list(range(np.size(input_matrix, 0)))
        self.layers = list([None] * np.size(topology))
        self.layers[0] = NeuronLayer(topology[0], input_matrix.size, bias)
        for i in range(1, len(topology)):
            self.layers[i] = NeuronLayer(topology[i], topology[i - 1], bias)

    def propagate_forward(self, input_matrix):
        self.layers[0].input = self.layers[0].weights * input_matrix + self.layers[0].bias
        self.layers[0].output = sigmoid(self.layers[0].input)
        for i in range(1, len(self.layers)):
            self.layers[i].input = self.layers[i].weights * self.layers[i - 1].output + self.layers[i].bias
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
