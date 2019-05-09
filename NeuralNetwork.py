import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NeuronLayer:
    def __init__(self, ileNeuronow, ileWejscnaNeuron):
        self.wagi = (2 * np.asmatrix(np.random.random((ileNeuronow, ileWejscnaNeuron))) - 1) / 2
        self.bias = np.asmatrix(np.zeros((ileNeuronow, 1)))
        self.input = np.asmatrix([])
        self.output = np.asmatrix([])


class NeuralNewtwork:
    def __init__(self, tablicaWarstw):
        self.layers = tablicaWarstw

    def propagate_forward(self, input_matrix):
        self.layers[0].input = self.layers[0].weights * input_matrix + self.layers[0].bias
        self.layers[0].output = sigmoid(self.layers[0].input)
        for i in range(1, self.layers.size):
            self.layers[i].input = self.layers[i].weights * self.layers[i - 1].output + self.layers[i].bias
            self.layers[i].output = sigmoid(self.layers[i].input)



nl = NeuronLayer(4, 2)
# sfs
print(nl.bias)
