import numpy as np
import pandas as pd
from subprocess import check_output
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
import prepData

#X_train, X_test, Y_train, Y_test = prepData.knn_iris()
X_train, X_test, Y_train, Y_test = prepData.knn_seeds()

# print("X_train shape: {}\ny_train shape: {}".format(X_train.shape, Y_train.shape))
# print("X_test shape: {}\ny_test shape: {}".format(X_test.shape, Y_test.shape))
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, Y_train)
Y_prediction = knn.predict(X_test)
output = pd.concat([X_test, Y_test, pd.Series(Y_prediction, name='Predicted', index=X_test.index)],
                   ignore_index=False, axis=1)
pickle.dump(output, open('KNN_output', 'wb'))
print(output)
print("Test set score: {:.2f}".format(knn.score(X_test, Y_test)))
