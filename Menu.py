import numpy as np
import NeuralNetwork as nn
import matplotlib.pyplot as plt
import pandas as pd
import pickle

topology = [2, 3, 4, 4]
bias = True

# file = pd.read_csv('seeds_dataset.txt', header=None)
file = pd.read_csv('iris.data', header=None)
file = file.drop(file.columns[4], axis=1)  # usuwa nazwy dla irysow
# file = file.drop(file.columns[7], axis=1) #dla ziaren
input_matrix = np.asmatrix(file.as_matrix())
df_height, df_width = input_matrix.shape

# train_data_ex2 = np.asmatrix([0, 1, 0, 0])
# train_data_ex1 = np.asmatrix([[1, 0, 0, 0],
#                               [0, 1, 0, 0],
#                               [0, 0, 1, 0],
#                               [0, 0, 0, 1]])
# df_height, df_width = train_data_ex1.shape
network = nn.NeuralNetwork(topology, bias, df_height)
moja_lambda = 0.5

# train
ox = list()
oy = list()
fig = plt.figure()
for i in range(1000):  # epoki
    cost = 0

    for x in range(df_height):
        network.propagate_back(input_matrix[x].T, input_matrix[x].T, moja_lambda, 0.7)
        # moja_lambda *= 0.99999
        # print(network.layers[-1].output)
        for q in range(df_height):
            cost += (network.layers[-1].error[q, 0] * network.layers[-1].error[q, 0])
    np.random.shuffle(input_matrix)
    ox.append(i)
    oy.append(float(cost))
plt.xlabel('Iteracja')
plt.ylabel('Błąd')
plt.plot(ox, oy)
plt.show()

# pickle.dump(network, open('perceptron', 'wb'))

# test
# network.propagate_forward(train_data_ex2.T)
#
# print(network.layers[-1].output)
#
#
# input_matrix = np.asmatrix([[1, 0, 0, 0],
#                               [0, 1, 0, 0],
#                               [0, 0, 1, 0],
#                               [0, 0, 0, 1]])

for i in range(df_height):
    network.propagate_forward( input_matrix[i].T)
    print("\n",  input_matrix[i], ": ")
    print(network.layers[-1].output)

# network = pickle.load(open('perceptron', 'rb'))
