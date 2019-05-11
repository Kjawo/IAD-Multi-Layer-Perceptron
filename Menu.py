import numpy as np
import NeuralNetwork as nn

topology = [2, 4]
bias = True

train_data_ex2 = np.asmatrix([0, 1, 0, 0])
train_data_ex1 = np.asmatrix([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
df_height, df_width = train_data_ex1.shape
network = nn.NeuralNetwork(topology, bias, df_height)
moja_lambda = 0.90
# train
for i in range(10000):  # epoki
    for x in range(df_height):
        network.propagate_back(train_data_ex1[x].T, train_data_ex1[x].T, moja_lambda, 0.9)
        moja_lambda *= 0.98
        # print(network.layers[-1].output)
    np.random.shuffle(train_data_ex1)

# test
network.propagate_forward(train_data_ex2.T)

print(network.layers[-1].output)
