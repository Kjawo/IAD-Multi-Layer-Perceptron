import pickle
import numpy as np

from PIL import Image
from matplotlib import image
from matplotlib import pyplot as plt


three = Image.open('digit3.png')
three.thumbnail((28, 28))
three = three.convert(mode='L')
three.save('digit3-prepared.png')

print(three.size)

data = image.imread('digit3-prepared.png')


# summarize shape of the pixel array
print(data.dtype)
print(data.shape)
# display the array of pixels as an image

plt.imshow(data, cmap="Greys")

data = data.reshape((1, 784))

network = pickle.load(open('minist-digits-test3', 'rb'))
network.propagate_forward(data.T)
guessed_class = np.where(network.layers[-1].output == np.amax(network.layers[-1].output))[0]
plt.title('Guessed: ' + str(guessed_class))
plt.show()
