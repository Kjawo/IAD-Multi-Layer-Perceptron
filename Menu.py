import numpy as np
import NeuralNetwork as nn
import matplotlib.pyplot as plt
import pickle

topology = [2, 4]
bias = True

train_data_ex2 = np.asmatrix([0, 1, 0, 0])
train_data_ex1 = np.asmatrix([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
df_height, df_width = train_data_ex1.shape
network = nn.NeuralNetwork(topology, bias, df_height)

_lambda = 0.9
_momentum = 0.0

# train
ox = list()
oy = list()
fig = plt.figure()
for i in range(100000):  # epoki
    cost = 0

    for x in range(df_height):
        network.propagate_back(train_data_ex1[x].T, train_data_ex1[x].T, _lambda, _momentum)
        # moja_lambda *= 0.99999
        # print(network.layers[-1].output)
        for q in range(df_height):
            cost += (network.layers[-1].error[q, 0] * network.layers[-1].error[q, 0])
    np.random.shuffle(train_data_ex1)
    ox.append(i)
    oy.append(float(cost))
    print(cost)
    if cost <= 0.001:
        print('Osiągnięto zadany błąd\nIteracja: ', i)
        break
plt.xlabel('Iteracja')
plt.ylabel('Błąd')
plt.plot(ox, oy)
plt.show()

# pickle.dump(network, open('perceptron', 'wb'))

train_data_ex1 = np.asmatrix([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

for i in range(df_height):
    network.propagate_forward(train_data_ex1[i].T)
    print("\n", train_data_ex1[i], ": ")
    print(network.layers[-1].output)

# network = pickle.load(open('perceptron', 'rb'))
