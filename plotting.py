import numpy as np
from matplotlib import pyplot as plt
import GPy
import functions
import kernels

"""
Loading nodes
"""

kernel = kernels.Trigonometric(input_dim=1, weight=0.5)
A = np.array([0, 1])
for i in range(2, 8):
    NoN = i
    h = 0.01
    name = kernel.__class__.__name__
    X = np.loadtxt("./Nodes/" + name + ", " + "{}".format(NoN) + ", " + "{}".format(h) + ".txt")
    Y = np.ones((int(round((1-h)/h, 0)), NoN))

    plt.figure()
    plt.grid()
    plt.title('Optimal node arrangement (NoN='+ '{}'.format(NoN) +')')
    plt.xlabel('Nodes')
    plt.ylabel('b-value')
    plt.xlim(A[0], A[1])
    for i in range(0, 99):
        plt.plot(X[i], h*i*Y[i], '.')
    plt.show()