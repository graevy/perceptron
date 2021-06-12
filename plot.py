# pylint: disable=unused-variable

import matplotlib.pyplot as plt
import numpy as np

# matplotlib.use('TkAgg')

# this needs to actually do linear algebra on data points
def plot(theta, theta_0=0, label=''):
    """graphs a perceptron iteration in 2 dimensions

    Args:
        theta (numpy array): a 2d numpy array
        theta_0 (int, optional): Defaults to 0.
    """

    # axis boundaries
    xmin = -5
    xmax = 5

    # formula for creating the line from both points:
    # (remember that numpy arrays are lists of lists, hence double indexing)
    # x*theta[0][0] + ytheta[0][1] = theta_0
    # simplifies to y = (theta_0 - x*theta[0][0]) / theta[0][1]

    # make two points by substituting values for x
    y1 = (theta_0 + xmin*theta[0][0])/theta[0][1]
    y2 = (theta_0 + xmax*theta[0][0])/theta[0][1]

    # feed the points into pyplot
    # right now this makes a new graph for each classifier iteration
    fig, ax = plt.subplots()
    ax.plot([xmin, xmax], [y1, y2], label=label)
