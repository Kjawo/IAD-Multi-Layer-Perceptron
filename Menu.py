import numpy as np
import NeuralNetwork as nn

topology = [3, 3, 1] #oznacza że sieć ma 3 warstwy kolejno po 3 3 i 1 neuron (nie licząc warstwy wejściowej)
bias = False
network = nn.NeuralNewtwork(topology,bias)
input_matrix = []
target_matrix = []
network.propagate_forward(input_matrix)
network.propagare_back(target_matrix)
print(len(topology))
