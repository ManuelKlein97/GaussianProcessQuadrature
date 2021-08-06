import numpy as np
import scipy as sp
import GPy
import kernels
import matplotlib
from matplotlib import pyplot as plt
import json
import scipy.stats as stats
from scipy.optimize import Bounds

def Optimal_Quadrature_Nodes_Optimizer(kernel: GPy.kern, number_of_nodes: int, initialguess: np.ndarray, return_var: bool):

    if (kernel.__class__.__name__ == 'PolynomialBasis'):
        def kernel_function(X, X2):
            return 1 / (1 - kernel.weight * X * X2)
        A = np.array([-1, 1])

    if (kernel.__class__.__name__ == 'Trigonometric'):
        def kernel_function(X, X2):
            up = 0.5 * (1 - np.square(kernel.weight))
            down = 1 + np.square(kernel.weight) - 2 * kernel.weight * np.cos(2 * np.pi * (X - X2))
            return 0.5 + up / down

        A = np.array([0, 1])

    if (kernel.__class__.__name__ == 'PolynomialBasisFinite'):
        def kernel_function(X, X2):
            n = 10
            upper = (1 - np.power((kernel.weight * X * X2), n + 1))
            lower = 1 - kernel.weight * X * X2
            return upper / lower

        A = np.array([-1, 1])

    def var(x: np.ndarray):
        n = len(x)
        u_X = np.zeros(n)
        K = np.identity(n)
        u_front_variance = sp.integrate.dblquad(kernel_function, A[0], A[1], A[0], A[1])[0]
        for i in range(0, n):
            u_X[i] = sp.integrate.quad(kernel_function, A[0], A[1], x[i])[0]
            for j in range(0, n):
                K[i, j] = kernel_function(x[i], x[j])
                K[j, i] = kernel_function(x[j], x[i])
        K_inv = np.linalg.pinv(K)
        temp = np.matmul(u_X, K_inv)
        for i in range(0, n):
            var = u_front_variance - np.matmul(temp, u_X)
        return var

    opt = sp.optimize.minimize(var, initialguess, method='Nelder-Mead')
    if (return_var == True):
        variance = var(opt.x)
        return opt.x, variance
    else:
        return opt.x