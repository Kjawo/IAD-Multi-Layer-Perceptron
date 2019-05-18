import numpy as np
import sns as sns

import NeuralNetwork as nn
import prepData
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import time


start = time.time()

# TODO: Refactor propagate_back
# TODO: Rename


topology = [6, 3]
bias = True
_lambda = 0.1
_momentum = 0.6
sciezka = 'encoder'

# train_input_matrix, train_target_matrix, test_input_matrix, test_target_matrix = prepData.iris()

train_input_matrix, train_target_matrix, test_input_matrix, test_target_matrix = prepData.iris()

nn.learn(1000, topology, train_input_matrix, train_target_matrix, _lambda, _momentum, bias, 1, 0.001, sciezka, False)
nn.test(test_input_matrix, test_target_matrix, topology, sciezka)

end = time.time()
print('\nExec time: ' + "%0.2f" % (end - start) + 's')
