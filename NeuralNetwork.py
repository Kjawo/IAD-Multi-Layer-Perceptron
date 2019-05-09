import numpy as np


def sigmoida(x):
    return 1 / (1 + np.exp(-x))


class NeuronLayer:
    def __init__(self, ileNeuronow, ileWejscnaNeuron):
        self.wagi = (2 * np.asmatrix(np.random.random((ileNeuronow, ileWejscnaNeuron))) - 1) / 2
        self.bias = np.asmatrix(np.zeros((ileNeuronow, 1)))
        self.input = np.asmatrix([])
        self.output = np.asmatrix([])


class NeuralNewtwork:
    def __init__(self, tablicaWarstw):
        self.layers = tablicaWarstw

    # def feed_forward(self, input):


nl = NeuronLayer(4, 2)
# sfs
print(nl.bias)
