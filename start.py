import matplotlib.pyplot as plt
import numpy as np

import perceptron


data = [
    [[1, -1], 1] ,
    [[0, 1], -1] ,
    [[-1.5, -1], 1]
]

plt.show()

print("Results! Theta = {}, theta_0 = {}".format(*perceptron.perceptron(data)))