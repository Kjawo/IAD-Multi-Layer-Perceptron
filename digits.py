import time

import NeuralNetwork as nn
import digits_functions

# import importlib
# importlib.reload(module)


start = time.time()

topology = [50, 10]
bias = True
_lambda = 0.3
_momentum = 0.6
# sciezka = 'minist-digits-hog-50'
sciezka = 'minist-digits-hog-50-test'

# train_input_matrix, train_target_matrix, test_input_matrix, test_target_matrix = prepData.iris()

train_X, train_Y, test_X, test_Y = digits_functions.digits_hog()
# train_X, train_Y, test_X, test_Y = digits_functions.digits()

nn.learn(15, topology, train_X, train_Y, test_X, test_Y, _lambda, _momentum, bias, 1, 0.001, sciezka, False, True)
nn.test(test_X, test_Y, topology, sciezka, False, False)

end = time.time()
print('\nExec time: ' + "%0.2f" % (end - start) + 's')
