import time

import NeuralNetwork as nn
import prepData

start = time.time()

topology = [6, 3]
bias = True
_lambda = 0.1
_momentum = 0.6
sciezka = 'test'

# train_input_matrix, train_target_matrix, test_input_matrix, test_target_matrix = prepData.iris()

train_input_matrix, train_target_matrix, test_input_matrix, test_target_matrix = prepData.iris()

nn.learn(100, topology, train_input_matrix, train_target_matrix, test_input_matrix, test_target_matrix, _lambda, _momentum, bias, 1, 0.001, sciezka, False, True)
nn.test(test_input_matrix, test_target_matrix, topology, sciezka, False, False)

end = time.time()
print('\nExec time: ' + "%0.2f" % (end - start) + 's')
