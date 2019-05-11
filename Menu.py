import numpy as np
import NeuralNetwork as nn

topology = [2, 4] #oznacza że sieć ma 3 warstwy kolejno po 3 3 i 1 neuron (nie licząc warstwy wejściowej)
bias = False

# train_data_ex1 = np.array([np.asmatrix([1, 0, 0, 0]).reshape(4, 1),
#                        np.asmatrix([0, 1, 0, 0]).reshape(4, 1),
#                        np.asmatrix([0, 0, 1, 0]).reshape(4, 1),
#                        np.asmatrix([0, 0, 0, 1]).reshape(4, 1)])
#
# train_xor = np.array([np.asmatrix([0, 0]).reshape(2, 1),
#                       np.asmatrix([1, 0]).reshape(2, 1),
#                       np.asmatrix([0, 1]).reshape(2, 1),
#                       np.asmatrix([1, 1]).reshape(2, 1)])

train_data_ex1 = np.array([np.asmatrix([1, 0, 0, 0]).reshape(4, 1),
                           np.asmatrix([0, 1, 0, 0]).reshape(4, 1),
                           np.asmatrix([0, 0, 1, 0]).reshape(4, 1),
                           np.asmatrix([0, 0, 0, 1]).reshape(4, 1)])



# corret_xor = np.vstack(np.array([0, 1, 1, 0]))

input_matrix = np.asmatrix([1, 0, 0, 0]).reshape(4, 1)

network = nn.NeuralNetwork(topology, bias, train_data_ex1[0])
# target_matrix = []

# network.propagate_forward(input_matrix)
# network.propagare_back(target_matrix)
# print(input_matrix)
# print(network.layers[0].weights)

# print(train_date)

moja_lambda = 0.4

# print(train_xor.shape)

# train
for i in range(600):  # epoki
    for x in range(train_data_ex1.shape[0]):
        network.propagate_back(train_data_ex1[x], train_data_ex1[x], moja_lambda, 0.6)
        moja_lambda *= 0.98
        # print(network.layers[-1].output)
    np.random.shuffle(train_data_ex1)

#test
network.propagate_forward(train_data_ex1[0])



print(network.layers[-1].output)

