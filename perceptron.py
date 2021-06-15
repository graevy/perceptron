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
# 2. perceptron calls iterate to evaluate a single data point
# 3. iterate calls evaluate, which determines if the classifier guessed the data's label correctly
# 4. perceptron updates the classifier with theta += y_i*x_i and theta_0 += y_i.
# iterate returns zeroes if the guess was accurate
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
    """determines whether or not the algorithm guessed properly

    Args:
        data_i (numpy array): data hyperpoint
        theta (numpy array): classifier hyperparameters
        theta_0 (float): 0d classifier parameter

    Returns:
        int: -1 if guessed incorrectly, 1 otherwise
    """
    # data_i[1] is y_i, the algorithm's guess.
    return data_i[1]*np.sign(np.dot(theta, data_i[0]) + theta_0)


def iterate(data_i, theta, theta_0):
    """pipes perceptron output to the classifier after evaluation

    Args:
        data_i (numpy array): data hyperpoint
        theta (numpy array): classifier hyperparameters
        theta_0 (float): 0d classifier parameter

    Returns:
        tuple: result of a perceptron iteration
    """
    if evaluate(data_i, theta, theta_0) <= 0:
        # adjustment necessary (return actual values)
        return (data_i, theta, theta_0)
    else:
        # no adjustment
        return False

def perceptron(data, theta=None, theta_0=0, t=1000):
    """introductory supervised machine learning algorithm to generate a linear classifier in n-dimensional space

    Args:
        data (numpy array): an entire dataset formatted [array, label], where label is 
        theta (numpy array, optional): classifier hyperparameters. Defaults to None.
        theta_0 (float, optional): 0d classifer parameter. Defaults to 0.
        t (int, optional): number of times to loop over data. Defaults to 1000.

    Returns:
        tuple: a tuple containing the resulting classifier (hyper)parameters
    """
    theta, theta_0 = np.zeros_like(data[0][0]), 0

    data, theta, theta_0 = format(data, theta, theta_0)

    index = 0
    # looping this makes classifier more accurate
    # for iteration in range(t):
    for point in data:
        index += 1

        # print(f"new data point: {point}")
        # print(f"{theta} is theta, {theta_0} is theta_0")
        result = iterate(point, theta, theta_0)

        if result != False:
            # theta += y_i * x_i; the algorithm updating itself
            # numpy doesn't like casting floats to ints, hence not "theta += result[0][1]*result[0][0]""
            np.add(theta, result[0][1]*result[0][0], out=theta, casting='unsafe')
            theta_0 += result[2]
            # print(f"{theta} is new theta, {theta_0} is new theta_0")

        # matplotlib.pyplot.plot unfortunately takes a 'label' kwarg to name each entry in a legend
        # do not confuse it with the labels attached to supervised learning data
        plot(theta, theta_0, label=str(index))


    return (theta, theta_0)
