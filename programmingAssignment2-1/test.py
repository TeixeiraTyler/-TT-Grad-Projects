import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from KNNObject import KNNObject


# Simple function to sum the amount correctly predicted and divide by total.
def accuracy(true, pred):
    return np.sum(true == pred) / len(true)


# Load in the data.
digits = load_digits()
X, y = digits.data, digits.target

# Split the data into train and test partitions.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1010)

# Perform KNN algorithm with given K value.
k = 10
clf = KNNObject(k)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

# Print resulting accuracy.
print("accuracy = ", accuracy(y_test, prediction))
