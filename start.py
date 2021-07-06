import matplotlib.pyplot as plt
import numpy as np

import perceptron


data = [
    [np.array([[1], [-1]]), 1] ,
    [np.array([[0], [1]]), -1] ,
    [np.array([[-10], [-1]]), 1]
]

print("Classifier: Theta.T = {.T}, theta_0 = {}".format(*perceptron.perceptron(data)))

plt.legend()
plt.title("Perceptron Classifiers")
plt.show()
