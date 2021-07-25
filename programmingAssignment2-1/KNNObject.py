import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# Custom object for storing variables and using custom functions for ease of use.
class KNNObject:

    # Initialize object with K value 5.
    def __init__(self, k=5):
        self.k = k

    # Insert X_train and y_train from test class.
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # Perform KNN algorithm on each point in X_test.
    def predict(self, X):
        return np.array([self.doKNN(x) for x in X])

    # The actual function for performing KNN.
    def doKNN(self, x):
        # Find distances between x and training data.
        distances = [l2Norm_distance(x, x_train) for x_train in self.X_train]

        # Sort distances and grab K-nearest neighbors.
        K_nearest = np.argsort(distances)[:self.k]

        # Store labels.
        labels = [self.y_train[i] for i in K_nearest]

        # Return the most common label.
        most_frequent = Counter(labels).most_common(1)
        return most_frequent[0][0]


# Simple function for calculating L2Norm (Euclidean) distance.
def l2Norm_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))
