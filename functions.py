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
get-functions and help-functions for a simpler reading of complex main-functions underneath
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

def get_weights(kernel: GPy.kern, X: np.ndarray, A: np.ndarray):

    kernel_function = get_kernel_function(kernel=kernel)
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

def get_weights_multi(kernel: GPy.kern, X: np.ndarray):
    '''

    :param kernel: kernel
    :param X: nodes
    :return: weights corresponding to the given kernel and nodes
    '''



    return 0

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
main-functions
'''

def Optimal_Quadrature_Nodes_Optimizer(kernel: GPy.kern, number_of_nodes: int, initialguess: np.ndarray, return_var: bool):
    '''
    Function to find the optimal node arrangement for a given kernel and a given number of nodes

    :param kernel: Working kernel
    :param number_of_nodes: Number of nodes wanted (not used)
    :param initialguess: Initialguesses implicitly delivering the number of nodes wanted (size of guesses)
    :param return_var: Bool wheather to return variance of resulting "optimal" nodes for comparison
    :return: Array of optimal nodes
    '''

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

def Optimizer_nodes_trigonometric(kernel: GPy.kern, number_of_nodes: int, opt_restarts: int, return_var: bool):
    '''
    Function to find the optimal node arrangement in 1D with the assumption that x_0 = 0 (Especially Trig. Kernel)

    :param kernel: Working kernel
    :param number_of_nodes: Number of nodes wanted
    :param opt_restarts: how often to restart the optimization process
    :param return_var: Bool wheather to return variance of resulting "optimal" nodes for comparison
    :return: Array of optimal nodes with x_0 = 0
    '''

    kernel_function = get_kernel_function(kernel=kernel)
    A = get_integration_area(kernel=kernel)

    def min_var_trig(x: np.ndarray):
        x = np.insert(x, 0, 0)
        n = len(x)
        u_X = np.ones(n)
        K = np.zeros((n, n))
        K[0, 0] = kernel_function(0, 0)
        for i in range(1, n):
            K[0, i] = kernel_function(0, x[i])
            K[i, 0] = kernel_function(x[i], 0)
            for j in range(1, n):
                K[i, j] = kernel_function(x[i], x[j])

        K_inv = np.linalg.pinv(K)
        temp = np.matmul(u_X, K_inv)
        var = 1 - np.matmul(temp, u_X)

        return var

    initialguess = np.random.uniform(low=A[0], high=A[1], size=(opt_restarts, number_of_nodes - 1))
    variances = np.ones(opt_restarts)
    possible_x = np.zeros((opt_restarts, number_of_nodes))

    for i in range(0, opt_restarts):
        opt = sp.optimize.minimize(min_var_trig, initialguess[i], method='Nelder-Mead')
        for j in range(0, len(opt.x)):
            possible_x[i][j] = opt.x[j]
        variances[i] = get_var(kernel=kernel, X=possible_x[i])

    optimal = possible_x[np.argmin(variances)]

    if (return_var == True):
        variance = min_var_trig(optimal)
        return optimal, variance
    else:
        return optimal

def Optimizer_nodes_2D(kernel: GPy.kern, initialguess: np.ndarray, return_var: bool):

    #initialguess also passes the number of nodes implicitly by its shape
    #Shape of initialguess should be n-2 for trigonometric and n for polynomial (-2 because of 2D)

    if kernel.__class__.__name__ == 'PolynomialBasis':
        def kernel_function(X1, X2, Y1, Y2):
            return 1/(1-kernel.weight*X1*Y1) * 1/(1-kernel.weight*X2*Y2)

        A = np.array([[-1, 1], [-1, 1]])

        def var(x: np.ndarray):
            n = int(0.5 * len(x))
            u_X = np.ones(n)
            K = np.zeros((n, n))
            range1 = np.arange(0, 2*n, 2)
            temp1 = np.zeros(n)
            for i in range1:
                u_X[int(i/2)] = 4/(np.square(kernel.weight)*x[i]*x[i+1]) * np.arctanh(kernel.weight * x[i]) * np.arctanh(kernel.weight * x[i+1])
                for j in range1:
                    K[int(i/2), int(j/2)] = kernel_function(x[i], x[i+1], x[j], x[j+1])

            u = np.power((2/kernel.weight * (sp.special.spence(1 - kernel.weight) - sp.special.spence(1 + kernel.weight))), n)
            K_inv = np.linalg.pinv(K)
            temp = np.matmul(u_X, K_inv)
            var = u - np.matmul(temp, u_X)

            return var

        opt = sp.optimize.minimize(var, initialguess, method='Nelder-Mead')

        if (return_var == True):
            variance = var(opt.x)
            return opt.x, variance
        else:
            return opt.x

    if kernel.__class__.__name__ == 'Trigonometric':
        def kernel_function(X1, X2, Y1, Y2):
            up = 0.5 * (1 - np.square(kernel.weight))
            down1 = 1 + np.square(kernel.weight) - 2 * kernel.weight * np.cos(2 * np.pi * (X1 - Y1))
            down2 = 1 + np.square(kernel.weight) - 2 * kernel.weight * np.cos(2 * np.pi * (X2 - Y2))
            return (0.5 + up / down1) * (0.5 + up / down2)

        A = np.array([[0, 1], [0, 1]])

        def var(x: np.ndarray):  #Shape of initialguess should be n-1 for this kernel
            x = np.insert(x, 0, 0) #inserting the zero point
            x = np.insert(x, 0, 0)
            n = int(0.5 * len(x))
            u_X = np.ones(n)
            K = np.zeros((n, n))
            K[0, 0] = kernel_function(0, 0, 0, 0)
            range1 = np.arange(2, 2*n, 2)
            for i in range1:
                K[0, int(i/2)] = kernel_function(0, 0, x[i], x[i+1])
                K[int(i/2), 0] = kernel_function(x[i], x[i+1], 0, 0)
                for j in range1:
                    K[int(i/2), int(j/2)] = kernel_function(x[i], x[i+1], x[j], x[j+1])

            K_inv = np.linalg.pinv(K)
            temp = np.matmul(u_X, K_inv)
            var = 1 - np.matmul(temp, u_X)

            return var

        opt = sp.optimize.minimize(var, initialguess, method='Nelder-Mead')
        optimal = np.insert(opt.x, 0, 0)
        optimal = np.insert(optimal, 0, 0)

        if (return_var == True):
            variance = var(optimal)
            return optimal, variance
        else:
            return optimal


#Quadrature rules

def GPQ(f, kernel: GPy.kern, A: np.ndarray, NoN: int, return_var: bool):
    '''
    Function to perform Gaussian process quadrature

    :param f: function to be integrated with GPQ
    :param kernel: kernel that shall be used for quadrature
    :param A: arbitrary integration area
    :param NoN: number of nodes involved
    :param return_var: Bool whether to return the variance of the predicted integral value
    :return: mean (, variance) of the posterior Gaussian process
    '''

    index = int(100 * kernel.weight - 1)
    kernel_function = get_kernel_function(kernel=kernel)
    A_std = get_integration_area(kernel=kernel)
    filename = kernel.__class__.__name__ + ', ' + '{}'.format(NoN) + ', ' + '{}'.format(0.01) + '.txt'
    X_std = np.loadtxt('./Nodes/' + filename)[index]
    X = integral_transformation(A=A_std, B=A, X=X_std)

    weights = get_weights(kernel=kernel, X=X, A=A)
    Y = f(X)

    I = np.matmul(weights, Y)
    if (return_var == True):
        u_X = np.zeros(NoN)
        for i in range(0, NoN):
            u_X[i] = sp.integrate.quad(kernel_function, A[0], A[1], X[i])[0]
        u = sp.integrate.dblquad(kernel_function, A[0], A[1], A[0], A[1])[0]
        var = u - np.matmul(weights, u_X)
        return I, var
    else:
        return I

def trapez(f, A: np.ndarray, NoN: int):
    '''
    Function to perform quadrature using the trapezoidal rule given a number of nodes

    :param f: function to integrate
    :param A: integration interval
    :param NoN: Number of nodes
    :return: Integral approximation value
    '''

    h = (A[1] - A[0]) / (NoN - 1)
    X = np.linspace(A[0], A[1], NoN)
    Y = f(X)
    W = np.array([0.5, 0.5])
    for i in range(2, NoN):
        W = np.insert(W, 1, 1)

    return h * np.dot(W, Y)

def GLQ(f, A: np.ndarray, NoN: int):
    '''
    Function to perform quadrature using the Gauss-Legendre quadrature rules given a number of nodes

    :param f: function to integrate
    :param A: integration interval
    :param NoN: Number of nodes
    :return: Integral approximation value
    '''
    nodes, weight = np.polynomial.legendre.leggauss(NoN)
    X = integral_transformation(A=np.array([-1, 1]), B=A, X=nodes)
    Y = f(X)
    h = (A[1] - A[0]) / 2

    return h * np.dot(weight, Y)

#Multidimensional Quadrature

def GPQ_multi(f, kernel: GPy.kern, A: np.ndarray, NoN: int, return_var: bool):

    if kernel.__class__.__name__ == 'PolynomialBasis':
        def kernel_function_to_int(X1: float, X2: float, Y1: float, Y2: float):
            return  1/(1 - kernel.weight * X1 * Y1) * 1/(1 - kernel.weight * X2 * Y2)

        def kernel_function(X: np.ndarray, X2: np.ndarray):
            if X.shape[1] == X2.shape[1]:
                input_dim = X.shape[1]
                n = X.shape[0]
                K = np.zeros((n, n))
                for i in range(0, n):
                    for j in range(0, n):
                        K[i, j] = kernel_function_to_int(X[i][0], X[i][1], X2[j][0], X2[j][1])

            else:
                raise AssertionError('Error, input dimensions of X and X2 do not match!')
            return K

    index = int(100 * kernel.weight - 1)
    filename = kernel.__class__.__name__ + ', ' + '{}'.format(NoN) + ', ' + '{}'.format(0.01) + '.txt'
    X_std = np.loadtxt('./Nodes/' + filename)[index]
    X = np.zeros((NoN ** 2, 2))
    NoN_2 = np.square(NoN)
    for i in range(0, len(X_std)):
        for j in range(0, NoN):
            X[i + j * NoN, 0] = X_std[j]
            X[i + j * NoN, 1] = X_std[i]

    Y_f = np.zeros(NoN_2)
    for i in range(0, NoN_2):
        Y_f[i] = f(X[i, 0], X[i, 1])

    u_X = np.zeros(NoN_2)
    for i in range(0, NoN_2):
        u_X[i] = sp.integrate.dblquad(kernel_function_to_int, A[0], A[1], A[0], A[1], [X[i][0], X[i][1]])[0]
    K = kernel_function(X, X)
    K_inv = np.linalg.pinv(K)
    weights = np.matmul(K_inv, u_X)
    I = np.matmul(weights, Y_f)
    if return_var == True:
        u = sp.integrate.nquad(kernel_function_to_int, [A, A, A, A])
        var = u - np.matmul(weights, u_X)
        return I, var
    else:
        return I

def GLQ_multi(f, A: np.ndarray, NoN: int):
    #A is not a nessesary input since this function is for the standart intervall only [-1, 1]
    nodes, weight = np.polynomial.legendre.leggauss(NoN)
    Y = np.zeros((NoN, NoN))
    sum = 0
    for i in range(0, NoN):
        for j in range(0, NoN):
            Y[i, j] = f(nodes[i], nodes[j])
            sum = sum + weight[i] * weight[j] * Y[i, j]

    return sum

#Sequential Bayesian Quadrature

def SBQ(kernel: GPy.kern, NoN_ex: int, opt_restarts: int, return_plot: bool):
    '''
    Function to perform one step of Sequential Bayesian Quadrature on the optimal nodes given

    :param kernel: Working kernel
    :param NoN_ex: Existing number of nodes used to load optimal nodes
    :param opt_restarts: Optimization restarts
    :return: Array of optimal nodes plus new "optimal" node from SBQ
    '''

    #Loading existing information
    index = int(100 * kernel.weight - 1)
    kernel_function = get_kernel_function(kernel=kernel)
    A = get_integration_area(kernel=kernel)
    filename = kernel.__class__.__name__ + ', ' + '{}'.format(NoN_ex) + ', ' + '{}'.format(0.01) + '.txt'
    X = np.loadtxt('./Nodes/' + filename)[index]

    def var_opt(x: np.ndarray):
        n = len(x)
        u_X = np.zeros(NoN_ex + n)
        K = np.identity(NoN_ex + n)
        u = sp.integrate.dblquad(kernel_function, A[0], A[1], A[0], A[1])[0]
        for i in range(0, NoN_ex):
            u_X[i] = sp.integrate.quad(kernel_function, A[0], A[1], X[i])[0]
            for j in range(0, NoN_ex):
                K[i, j] = kernel_function(X[i], X[j])
                K[j, i] = kernel_function(X[j], X[i])

        for i in range(0, n):
            u_X[NoN_ex + i] = sp.integrate.quad(kernel_function, A[0], A[1], x[i])[0]
            for j in range(0, NoN_ex):
                K[j, NoN_ex + i] = kernel_function(X[j], x[i])
                K[NoN_ex + i, j] = kernel_function(x[i], X[j])
            K[NoN_ex, NoN_ex] = kernel_function(x[i], x[i])


        K_inv = np.linalg.pinv(K)
        temp = np.matmul(u_X, K_inv)
        vari = u - np.matmul(temp, u_X)
        return vari

    if return_plot == True:

        n = 100
        X_lin = np.linspace(A[0], A[1], n)
        Y_lin = np.zeros(n)
        for i in range(0, n):
            Y_lin[i] = var_opt(np.array([X_lin[i]]))

        plt.figure()
        plt.grid()
        if np.max(Y_lin) < 0.001: plt.yscale('log')
        plt.xlim(A[0], A[1])
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.plot(X_lin, Y_lin)
        plt.show()

    #Minimization
    nodes = np.zeros(NoN_ex + 1)
    for i in range(0, NoN_ex):
        nodes[i] = X[i]

    initialguess = np.random.uniform(low=A[0], high=A[1], size=opt_restarts)
    variances = np.ones(opt_restarts)
    possible_x = np.zeros(opt_restarts)
    for i in range(0, opt_restarts):
        opt = sp.optimize.minimize(var_opt, initialguess[i], method='Nelder-Mead')
        possible_x[i] = opt.x
        nodes[NoN_ex] = opt.x
        variances[i] = get_var(kernel=kernel, X=nodes)

    best_x = possible_x[np.argmin(variances)]
    nodes[NoN_ex] = best_x

    return nodes

def SBQ_X(kernel: GPy.kern, X: np.ndarray, opt_restarts: int, return_plot: bool):
    '''
    Function to perform one step of Sequential Bayesian Quadrature given a set of nodes X

    :param kernel: Working kernel
    :param X: Existing nodes
    :param opt_restarts: Restarts of the optimization
    :return: New set of nodes X union x_new (SBQ)
    '''

    kernel_function = get_kernel_function(kernel=kernel)
    A = get_integration_area(kernel=kernel)
    NoN_ex = len(X)


    def var_opt(x: np.ndarray):
        n = len(x)
        u_X = np.zeros(NoN_ex + n)
        K = np.identity(NoN_ex + n)
        u = sp.integrate.dblquad(kernel_function, A[0], A[1], A[0], A[1])[0]
        for i in range(0, NoN_ex):
            u_X[i] = sp.integrate.quad(kernel_function, A[0], A[1], X[i])[0]
            for j in range(0, NoN_ex):
                K[i, j] = kernel_function(X[i], X[j])
                K[j, i] = kernel_function(X[j], X[i])

        for i in range(0, n):
            u_X[NoN_ex + i] = sp.integrate.quad(kernel_function, A[0], A[1], x[i])[0]
            for j in range(0, NoN_ex):
                K[j, NoN_ex + i] = kernel_function(X[j], x[i])
                K[NoN_ex + i, j] = kernel_function(x[i], X[j])
            K[NoN_ex, NoN_ex] = kernel_function(x[i], x[i])


        K_inv = np.linalg.pinv(K)
        temp = np.matmul(u_X, K_inv)
        vari = u - np.matmul(temp, u_X)
        return vari

    if return_plot == True:

        n = 200
        X_lin = np.linspace(A[0], A[1], n)
        Y_lin = np.zeros(n)

        for i in range(0, n):
            Y_lin[i] = var_opt(np.array([X_lin[i]]))

        plt.figure()
        plt.grid()
        if np.max(Y_lin) < 0.001: plt.yscale('log')
        plt.xlim(A[0], A[1])
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.plot(X_lin, Y_lin)
        plt.show()

    #Minimization
    nodes = np.zeros(NoN_ex + 1)
    for i in range(0, NoN_ex):
        nodes[i] = X[i]

    initialguess = np.random.uniform(low=A[0], high=A[1], size=opt_restarts)
    variances = np.ones(opt_restarts)
    possible_x = np.zeros(opt_restarts)
    for i in range(0, opt_restarts):
        opt = sp.optimize.minimize(var_opt, initialguess[i], method='Nelder-Mead')
        possible_x[i] = opt.x
        nodes[NoN_ex] = opt.x
        variances[i] = get_var(kernel=kernel, X=nodes)

    best_x = possible_x[np.argmin(variances)]
    nodes[NoN_ex] = best_x

    return nodes