import pickle
import numpy as np
import digits_functions

from PIL import Image
from matplotlib import image
from matplotlib import pyplot as plt

hog = True

if hog:
    digit, hog_of_digit = digits_functions.prep_image_hog('digit3.png')
else:
    digit = digits_functions.prep_image('digit3.png')

plt.imshow(digit, cmap="Greys")

digit = digit.reshape((1, 784))

if hog:
    network_path = 'minist-digits-hog-50'
else:
    network_path = 'minist-digits-test3'

network = pickle.load(open(network_path, 'rb'))

if hog:
    network.propagate_forward(hog_of_digit.reshape(36, 1))
else:
    network.propagate_forward(digit.T)

guessed_class = np.where(network.layers[-1].output == np.amax(network.layers[-1].output))[0]
plt.title('Guessed: ' + str(guessed_class))

plt.show()
