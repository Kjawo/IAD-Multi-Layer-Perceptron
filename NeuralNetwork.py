import numpy as np


def sigmoida(x):
    return 1 / (1 + np.power(np.e, x))


class NeuronLayer:
    def __init__(self, ileNeuronow, ileWejscnaNeuron):
        self.wagi = (2 * np.asmatrix(np.random.random((ileNeuronow, ileWejscnaNeuron))) - 1) / 2


nl = NeuronLayer(4, 2)

print(nl.wagi)