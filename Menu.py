import numpy as np
import NeuralNetwork as nn
import matplotlib.pyplot as plt
import pickle
import pandas as pd

topology = [4, 3]
bias = False

encoder = np.asmatrix([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

xor = np.asmatrix([[1, 0],
                   [0, 1],
                   [0, 0],
                   [1, 1]])

xor_correct = np.asmatrix([[1],
                   [1],
                   [0],
                   [0]])


file = pd.read_csv('iris.data', header=None)

output = file[4]


# _target_matrix = np.zeros((file.shape[0], 3))
#
# for row in range(output.size):
#     _target_matrix[row][output[row]] = 1

_target_matrix = np.zeros((file.shape[0], 1))

for row in range(output.size):
    _target_matrix[row] = output[row]

file = file.drop(columns=[4])
input_matrix = np.asmatrix(file.as_matrix())

# input_matrix = xor
# _target_matrix = xor_correct



df_height, df_width = input_matrix.shape
# network = nn.NeuralNetwork(topology, bias, df_height)

_lambda = 0.1
_momentum = 0.1
sciezka = 'irysy2'

nn.learn(10000, topology, input_matrix, _target_matrix, _lambda, _momentum, bias, sciezka)
# train
# ox = list()
# oy = list()
# fig = plt.figure()
# for i in range(100000):  # epoki
#     cost = 0
#
#     for x in range(df_height):
#         network.propagate_back(train_data_ex1[x].T, train_data_ex1[x].T, _lambda, _momentum)
#         # moja_lambda *= 0.99999
#         # print(network.layers[-1].output)
#         for q in range(df_height):
#             cost += (network.layers[-1].error[q, 0] * network.layers[-1].error[q, 0])
#     np.random.shuffle(train_data_ex1)
#     ox.append(i)
#     oy.append(float(cost))
#     print(cost)
#     if cost <= 0.001:
#         print('Osiągnięto zadany błąd\nIteracja: ', i)
#         break
# plt.xlabel('Iteracja')
# plt.ylabel('Błąd')
# plt.plot(ox, oy)
# plt.show()

# pickle.dump(network, open('perceptron', 'wb'))


network = pickle.load(open(sciezka, 'rb'))
# train_data_ex1 = np.asmatrix([[1, 0, 0, 0],
#                               [0, 1, 0, 0],
#                               [0, 0, 1, 0],
#                               [0, 0, 0, 1]])


np.set_printoptions(suppress=True)
for i in range(df_height):
    network.propagate_forward(input_matrix[i].T)
    print("\n", input_matrix[i], ": ")
    print(network.layers[-1].output)

# network = pickle.load(open('perceptron', 'rb'))