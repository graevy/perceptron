import matplotlib.pyplot as plt
import numpy as np


def plot(theta, theta_0=0, label=''):
    """graphs a perceptron iteration in 2 dimensions

    Args:
        theta (numpy array): a 2d numpy array
        theta_0 (int, optional): Defaults to 0.
        label (string, optional): add each perceptron iteration to a graph legend. Defaults to ''
    """

    # axis boundaries
    xmin = -5
    xmax = 5

    # formula for creating the line from both points:
    # (remember that numpy arrays are lists of lists, hence double indexing)
    # x*theta[0][0] + y*theta[0][1] = -theta_0
    # simplifies to y = (theta_0 - x*theta[0][0]) / theta[1][0]

    # make two points by substituting values for x
    y1 = -(theta_0 + xmin*theta[0][0])/theta[1][0]
    y2 = -(theta_0 + xmax*theta[0][0])/theta[1][0]

    # feed the points into pyplot
    plt.plot([xmin, xmax], [y1, y2], label=label)

##########################################
# A better but untested implementation
##########################################
def bounds(data, extra=True):
    xmin, xmax = 0, 0

    for item in data:
        if item[0][0] < xmin:
            xmin = item[0][0]
        elif item[0][0] > xmax:
            xmax = item[0][0]

    if extra:
        return (xmin-1, xmax+1)
    else:
        return (xmin, xmax)


# TODO
def ndplot(theta, theta_0=0, label='', data=None):
    if data != None:
        xmin, xmax = bounds(data)
        dimensions = len(data[0][0])
    else:
        xmin, xmax = -5, 5
        dimensions = 2


    if dimensions == 2:
        xvalues = np.linspace(xmin, xmax)
        yvalues = []

        # for each x value in xvalues, θ.T • X = -θ_0
        for x in xvalues:
            # solve for the y coordinate where (θ.T)•X = -θ_0
            # θ = [a,b], coords = [x,y]
            # so ax + by = -θ_0,
            # y = -(θ_0 + ax) / b
            # plot it with pyplot:
            yvalues.append(  -(theta_0 + theta[0][0]*x) / theta[1][0]  )
        
        plt.plot(xvalues, yvalues, label=label)

    else:
        pass # TODO
        # 3d seems much trickier, but really you just have to call 2d plot on a series of lines to get a plane
        # this lends itself well to nd recursion (e.g. yielding 3d spaces for a 4d theta)
        # dot product of 2d vectors [a,b] and [c,d] is ac + bd
        # dot product of 3d vectors [a,b,c] and [d,e,f] is ad + be + cf
        # dot product of nd vectors va: [a, ..., n] and vb: [b, ..., n] is:
        # sum([va[index] * vb[index] for index in range(n)])
        # Theta.T • X + Theta_0 = 0
        # Theta.T • X = -Theta_0
        # n = len(theta)