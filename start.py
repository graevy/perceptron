import matplotlib.pyplot as plt
import numpy as np

import perceptron


data = [
    [np.array([[0], [1]]), -1] ,
    [np.array([[-1.5], [-1]]), 1] ,
    [np.array([[1], [-1]]), 1]
]

print("Results! Theta = {}, theta_0 = {}".format(*perceptron.perceptron(data)))

plt.legend()
plt.title("Perceptron")
plt.show()
