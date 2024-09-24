#!./getting_data/bin/python

import nnfs
from nnfs.datasets import spiral_data

nnfs.init()
X, y = spiral_data(samples=100, classes=2)
X_test, y_test = spiral_data(samples=100, classes=2)

print(X)
print("----------------------------------------")
print(X_test)
print("----------------------------------------")
print(y)
print("----------------------------------------")
print(y_test)
