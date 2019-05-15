import numpy as np
import NeuralNetwork as nn
import prepData
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import time

start = time.time()

# TODO: Klasyfikator!
# TODO: Refactor propagate_back
# TODO: Rename
# TODO: User interface


topology = [6, 3]
bias = True
_lambda = 0.6
_momentum = 0.1
sciezka = 'encoder'

# train_input_matrix, train_target_matrix, test_input_matrix, test_target_matrix = prepData.iris()
train_input_matrix, train_target_matrix, test_input_matrix, test_target_matrix = prepData.seeds()

nn.learn(100, topology, train_input_matrix, train_target_matrix, _lambda, _momentum, bias, 1, 0.001, sciezka, False)

nn.test(test_input_matrix, test_target_matrix, topology, sciezka)

end = time.time()
print('\nExec time: ' + "%0.2f" % (end - start) + 's')
