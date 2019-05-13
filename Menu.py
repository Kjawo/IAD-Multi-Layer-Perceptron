import numpy as np
import NeuralNetwork as nn
import prepData
import matplotlib.pyplot as plt
import pickle
import pandas as pd

topology = [8, 1]
bias = False
_lambda = 0.6
_momentum = 0.0
sciezka = 'irysy2'

input_matrix, target_matrix = prepData.iris()

nn.learn(1000, topology, input_matrix, target_matrix, _lambda, _momentum, bias, sciezka)

nn.test(input_matrix, target_matrix, sciezka)