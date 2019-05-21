import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import digits_functions
import NeuralNetwork as nn
import prepData

# import importlib
# importlib.reload(module)


start = time.time()

topology = [50, 10]
bias = True
_lambda = 0.3
_momentum = 0.6
sciezka = 'minist-digits-test2'

# train_input_matrix, train_target_matrix, test_input_matrix, test_target_matrix = prepData.iris()

train_input_matrix, train_target_matrix, test_input_matrix, test_target_matrix = digits_functions.digits()

# nn.learn(1, topology, train_input_matrix, train_target_matrix, _lambda, _momentum, bias, 1, 0.001, sciezka, False)
nn.test(test_input_matrix, test_target_matrix, topology, sciezka, False, True)

end = time.time()
print('\nExec time: ' + "%0.2f" % (end - start) + 's')