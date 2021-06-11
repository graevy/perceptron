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

def format(data, theta, theta_0):
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
    # data_i[1] is y_i, the algorithm's guess.
    # print("dotting {} and {}".format(theta, data_i[0]))
    return data_i[1]*np.sign(np.dot(theta, data_i[0]) + theta_0)


def iterate(data_i, theta, theta_0):
    if evaluate(data_i, theta, theta_0) <= 0:
        print("bad guess")
        # adjustment necessary (return actual values)
        return (data_i, theta, theta_0)
    else:
        print("good guess")
        # no adjustment (everything is zero)
        return (np.zeros_like(data_i[0]), 0, 0)

def perceptron(data, theta=None, theta_0=0, t=1000):
    # dimension = len(data[0])
    theta, theta_0 = np.zeros_like(data[0][0]), 0

    # print("before formatting: data="+str(data)+', theta='+str(theta)+', theta_0='+str(theta_0))
    data, theta, theta_0 = format(data, theta, theta_0)
    # print("after formatting: data="+str(data)+', theta='+str(theta)+', theta_0='+str(theta_0))

    index = 0
    # looping this makes classifier more accurate
    # for iteration in range(t):
    for point in data:
        index += 1

        print("new data point: {}".format(point))
        print("{} is theta, {} is theta_0".format(theta, theta_0))
        result = iterate(point, theta, theta_0)

        # TODO: incomplete function
        plot(point, theta_0, label=str(index))

        # theta += y_i * x_i
        # theta += result[0][1]*result[0][0] doesn't work because numpy doesn't like casting floats to ints
        np.add(theta, result[0][1]*result[0][0], out=theta, casting='unsafe')
        theta_0 += result[2]
        print("{} is new theta, {} is new theta_0".format(theta, theta_0))

    return (theta, theta_0)
