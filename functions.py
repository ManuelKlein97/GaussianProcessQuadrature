import numpy as np
import scipy as sp
import GPy
import kernels
import matplotlib
from matplotlib import pyplot as plt
import json
import scipy.stats as stats
from scipy.optimize import Bounds

'''
get-functions for a more simple reading of complex functions underneath
'''

def get_kernel_function(kernel: GPy.kern):

    if (kernel.__class__.__name__ == 'PolynomialBasis'):
        def kernel_function(X, X2):
            return 1 / (1 - kernel.weight * X * X2)

    if (kernel.__class__.__name__ == 'Trigonometric'):
        def kernel_function(X, X2):
            up = 0.5 * (1 - np.square(kernel.weight))
            down = 1 + np.square(kernel.weight) - 2 * kernel.weight * np.cos(2 * np.pi * (X - X2))
            return 0.5 + up / down

    if (kernel.__class__.__name__ == 'PolynomialBasisFinite'):
        def kernel_function(X, X2):
            n = 8
            upper = (1 - np.power((kernel.weight * X * X2), n + 1))
            lower = 1 - kernel.weight * X * X2
            return upper / lower

    return kernel_function

def get_integration_area(kernel: GPy.kern):
    if (kernel.__class__.__name__ == 'PolynomialBasis'):
        A = np.array([-1, 1])

    if (kernel.__class__.__name__ == 'Trigonometric'):
        A = np.array([0, 1])

    if (kernel.__class__.__name__ == 'PolynomialBasisFinite'):
        A = np.array([-1, 1])

    return A

def get_var(kernel: GPy.kern, X: np.ndarray):

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
            n = 8
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
        K_inv = np.linalg.pinv(K)
        temp = np.matmul(u_X, K_inv)
        var = u_front_variance - np.matmul(temp, u_X)
        return var

    return var(X)

def get_weights(kernel: GPy.kern, X: np.ndarray):

    kernel_function = get_kernel_function(kernel=kernel)
    A = get_integration_area(kernel=kernel)

    n = len(X)
    u_X = np.zeros(n)
    K = np.identity(n)

    for i in range(0, n):
        u_X[i] = sp.integrate.quad(kernel_function, A[0], A[1], X[i])[0]
        for j in range(0, n):
            K[i, j] = kernel_function(X[i], X[j])

    K_inverse = np.linalg.pinv(K)
    weights = np.matmul(K_inverse, u_X)

    return weights

def integral_transformation(A: np.ndarray, B: np.ndarray, X: np.ndarray):
    '''

    :param A: First integration area (standard)
    :param B: Target integration area
    :param X: Points that are to be transformed
    :return: Transformed points Y
    '''

    n = len(X)
    Y = np.zeros(n)
    for i in range(0, n):
        Y[i] = (X[i] - A[0])/(A[1] - A[0]) * B[1] + (A[1] - X[i])/(A[1] - A[0]) * B[0]

    return Y

'''
Main functions
'''

def Optimal_Quadrature_Nodes_Optimizer(kernel: GPy.kern, number_of_nodes: int, initialguess: np.ndarray, return_var: bool):

    kernel_function = get_kernel_function(kernel=kernel)
    A = get_integration_area(kernel=kernel)

    def var(x: np.ndarray):
        n = len(x)
        u_X = np.zeros(n)
        K = np.identity(n)
        u_front_variance = sp.integrate.dblquad(kernel_function, A[0], A[1], A[0], A[1])[0]
        for i in range(0, n):
            u_X[i] = sp.integrate.quad(kernel_function, A[0], A[1], x[i])[0]
            for j in range(0, n):
                K[i, j] = kernel_function(x[i], x[j])
        K_inv = np.linalg.pinv(K)
        temp = np.matmul(u_X, K_inv)
        #for i in range(0, n):
        vari = u_front_variance - np.matmul(temp, u_X)
        return vari

    opt = sp.optimize.minimize(var, initialguess, method='Nelder-Mead')
    if (return_var == True):
        variance = var(opt.x)
        return opt.x, variance
    else:
        return opt.x

def Optimal_Quadrature_Nodes_Extend(kernel: GPy.kern, NoN_ex: int, NoN_add: int, return_var: bool):

    '''

    :param kernel:
    :param NoN_ex: Already existing Number of nodes
    :param NoN_add: Number of nodes wanted to be added
    :return:
    '''

    filename = kernel.__class__.__name__ + ', ' + '{}'.format(NoN_ex) + ', ' + '{}'.format(0.01) + '.txt'
    X = np.loadtxt(filename)
    K = np.zeros((NoN_ex, NoN_ex))
    u_X = np.zeros(NoN_ex)
    kernel_function = get_kernel_function(kernel=kernel)
    A = get_integration_area(kernel=kernel)

    #Generating existing Vectors and Matrices
    u_front_variance = sp.integrate.dblquad(kernel_function, A[0], A[1], A[0], A[1])[0]
    for i in range(0, NoN_ex):
        u_X[i] = sp.integrate.quad(kernel_function, A[0], A[1], X[i])[0]
        for j in range(0, NoN_ex):
            K[i, j] = kernel_function(X[i], X[j])
            K[j, i] = kernel_function(X[j], X[i])

    def extended_variance(x: np.ndarray):
        n = len(x)
        variance = np.zeros(n)
        for k in range(0, n):
            u_X[NoN_ex + NoN_add - 1] = sp.integrate.quad(kernel_function, A[0], A[1], x[k])[0]
            for i in range(0, NoN_ex + NoN_add):
                K[NoN_ex + NoN_add - 1, i] = kernel_function(X[i], x[k])
                K[i, NoN_ex + NoN_add - 1] = kernel_function(x[k], X[i])
            K_inv = np.linalg.pinv(K)
            temp = np.matmul(u_X, K_inv)
            variance[k] = u_front_variance - np.matmul(temp, u_X)
        return variance

    initialguess = np.random.uniform(low=A[0], high=A[1], size=NoN_add)
    opt = sp.optimize.minimize(extended_variance, initialguess, method='Nelder-Mead', bounds=((A[0], A[1])))
    variance = get_var(kernel=kernel, X=opt.x)
    if return_var == True:
        return opt.x, variance
    else:
        return opt.x

def GPQ(f, kernel: GPy.kern, A: np.ndarray, NoN: int, return_var: bool):
    '''

    :param f: function to be integrated with GPQ
    :param kernel: kernel that shall be used for quadrature
    :param A: arbitrary integration area
    :param NoN: number of nodes involved
    :param return_var: Bool whether to return the variance of the predicted integral value
    :return:
    '''

    kernel_function = get_kernel_function(kernel=kernel)
    A_std = get_integration_area(kernel=kernel)
    filename = kernel.__class__.__name__ + ', ' + '{}'.format(NoN) + ', ' + '{}'.format(0.01) + '.txt'
    X_std = np.loadtxt(filename)
    X = integral_transformation(A=A_std, B=A, X=X_std)

    u_X = np.zeros(NoN)
    K = np.zeros((NoN, NoN))
    Y = f(X)

    for i in range(0, NoN):
        u_X[i] = sp.integrate.quad(kernel_function, A[0], A[1], X[i])[0]
        for j in range(0, NoN):
            K[i, j] = kernel_function(X[i], X[j])

    K_inv = np.linalg.pinv(K)
    temp1 = np.matmul(u_X, K_inv)
    I = np.matmul(temp1, Y)
    if (return_var == True):
        u = sp.integrate.dblquad(kernel_function, A[0], A[1], A[0], A[1])[0]
        var = u - np.matmul(temp1, u_X)
        return I, var
    else:
        return I