# pylint: disable=unused-variable

import numpy as np
from plot import plot

##########################################################
# THE PERCEPTRON ALGORITHM
# perceptron(data)
# theta = [[zeroes]], theta_0 = 0
# for point in data:
#   if y_i*(np.dot(theta.T, x_i) + theta_0) <= 0:
#       theta += y_i*x_i
#       theta_0 += y_i
# the conditional triggers when we guess incorrectly
#
# if you iterate over data infinitely,
# eventually you WILL have a perfect classifier
##########################################################

##########################################################
# Program Layout
# 1. perceptron calls format to make sure the data is compatible with the algorithm
# 2. perceptron calls evaluate
# 3. evaluate determines if the classifier guessed the data's label correctly
# 4. evaluate returns relevant data to update classifier parameters with
# 5. perceptron updates the classifier with theta += y_i*x_i and theta_0 += y_i.
##########################################################


def format(data, theta, theta_0):
    """formats a dataset to be compatible with the perceptron algorithm

    Args:
        data (a list or numpy array): the whole dataset
        theta (a numpy array): hyperparameters for classification
        theta_0 (float): parameter

    Returns:
        tuple: the input, formatted for evaluation
    """
    # to output a scalar via np.dot(theta, x)
    # theta must be a 1 by d array
    theta = np.array([theta])
    if theta.shape[0] != 1:
        print("theta shape error")
    
    if type(theta_0) is not int:
        print("theta_0 type error")

    for data_i in data:
        # x_i must be a d by 1 array
        x_i = np.array([data_i[0]])
        y_i = data_i[1]
        data_i = [x_i.T, y_i]

        if type(data_i[1]) is not int:
            print("y_i type error")

    return (data, theta, theta_0)

def evaluate(data_i, theta, theta_0):
    """determines whether or not the classifier was correct, and returns relevant data

    Args:
        data_i (numpy array): data hyperpoint
        theta (numpy array): classifier hyperparameters
        theta_0 (float): 0d classifier parameter

    Returns:
        tuple or False: interpreted by perceptron
    """
    # data_i[0] is x_i, the hyperpoint
    # data_i[1] is y_i, the label for that hyperpoint (-1 or 1)
    guess = data_i[1]*np.sign(np.dot(theta, data_i[0]) + theta_0)

    if guess <= 0:
        return (data_i, theta, theta_0)
    else:
        return False


def perceptron(data, theta=None, theta_0=0, t=1):
    """introductory supervised machine learning algorithm to generate a linear classifier in n-dimensional space

    Args:
        data (numpy array): an entire dataset formatted [array, label], where label is 
        theta (numpy array, optional): classifier hyperparameters. Defaults to None.
        theta_0 (float, optional): 0d classifer parameter. Defaults to 0.
        t (int, optional): number of times to loop over data. Defaults to 1000.

    Returns:
        tuple: a tuple containing the resulting classifier (hyper)parameters
    """

    if theta == None:
        theta, theta_0 = np.zeros_like(data[0][0]), 0

    data, theta, theta_0 = format(data, theta, theta_0)

    index = 0

    # looping this makes classifier more accurate
    for iteration in range(t):
        for point in data:

            index += 1
            print(f"Loop number {index}")
            print(f"{theta} is theta, {theta_0} is theta_0")
            print(f"new data point. y_i: {point[1]} x_i {point[0]}")


            result = evaluate(point, theta, theta_0)

            if result != False:
                # theta += y_i * x_i; the algorithm updating itself
                # yixi is y_i * x_i
                yixi = np.multiply(result[0][1], result[0][0])
                # numpy doesn't like casting floats to ints, hence not "theta += result[0][1]*result[0][0]""
                np.add(theta, yixi, out=theta, casting='unsafe')
                theta_0 += result[0][1]

            # matplotlib.pyplot.plot unfortunately takes a 'label' kwarg to name each entry in a legend
            # do not confuse it with the labels attached to supervised learning data
            plot(theta, theta_0, label=str(index))


    return (theta, theta_0)
