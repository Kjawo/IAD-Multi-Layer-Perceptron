import numpy as np
import NeuralNetwork as nn
import prepData
import matplotlib.pyplot as plt
import pickle
import pandas as pd

# TODO: Klasyfikator!
# TODO: Refactor propagate_back
# TODO: Rename
# TODO: Add seed data
# TODO: User interface


topology = [6, 3]
bias = True
_lambda = 0.6
_momentum = 0.0
sciezka = 'encoder'

train_input_matrix, train_target_matrix, test_input_matrix, test_target_matrix = prepData.iris()

nn.learn(1000, topology, train_input_matrix, train_target_matrix, _lambda, _momentum, bias, 0.001, sciezka)

nn.test(test_input_matrix, test_target_matrix, topology, sciezka)