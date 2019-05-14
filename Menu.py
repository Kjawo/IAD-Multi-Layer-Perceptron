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


topology = [2, 4]
bias = True
_lambda = 0.6
_momentum = 0.0
sciezka = 'encoder'

input_matrix, target_matrix = prepData.encoder()

nn.learn(500, topology, input_matrix, target_matrix, _lambda, _momentum, bias, 0.001, sciezka)

nn.test(input_matrix, target_matrix, topology, sciezka)