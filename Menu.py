import time

import NeuralNetwork as nn
import prepData

import matplotlib.pyplot as plt

start = time.time()

topology = [16, 1]
bias = True
_lambda = 0.3
_momentum = 0.6
sciezka = 'test'

# train_input_matrix, train_target_matrix, test_input_matrix, test_target_matrix = prepData.iris()

# train_X, train_Y, test_X, test_Y = prepData.seeds()
#
# nn.learn(100, topology, train_X, train_Y, test_X, test_Y, _lambda, _momentum, bias, 1, 0.001, sciezka, False, True)
# nn.test(test_X, test_Y, topology, sciezka, False, False)

train_X, train_Y, test_X, test_Y = prepData.sin_data(True)
plt.plot(test_X, test_Y)
plt.show()

nn.learn(10, topology, train_X, train_Y, test_X, test_Y, _lambda, _momentum, bias, 1, 0.001, sciezka, False, True)
nn.test(test_X, test_Y, topology, sciezka, False, False)

end = time.time()
print('\nExec time: ' + "%0.2f" % (end - start) + 's')
