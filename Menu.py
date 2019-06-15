import time

import NeuralNetwork as nn
import prepData
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

start = time.time()

topology = [64, 1]
rbf_topology = [True, False]
bias = True
_lambda = 0.3
_momentum = 0.6
sciezka = 'test'

# train_X, train_Y, test_X, test_Y = prepData.iris()

# train_X, train_Y, test_X, test_Y = prepData.seeds()
#
# nn.learn(100, topology, train_X, train_Y, test_X, test_Y, _lambda, _momentum, bias, 1, 0.001, sciezka, False, True)
# nn.test(test_X, test_Y, topology, sciezka, False, False)

train_X, train_Y, test_X, test_Y = prepData.sin_cos_data(True)

# plt.plot(test_X[:, 0], test_X[:, 1], test_Y)
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(test_X[:, 0], test_X[:, 1], test_Y)
# plt.show()
#
nn.learn(10, topology, train_X, train_Y, test_X, test_Y, _lambda, _momentum, bias, 1, 0.001, sciezka, False, True,
         rbf_topology)
nn.test(test_X, test_Y, topology, sciezka, False, False, train_X.shape[1])

end = time.time()
print('\nExec time: ' + "%0.2f" % (end - start) + 's')
