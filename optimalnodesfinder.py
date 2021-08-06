import kernels
import functions
import numpy as np

'''
This python file is a template for finding the optimal nodes for a given kernel.
The saving data template is as follows:
"Kernelname, NumberOfNodes, h-value.txt"
'''

'''
For kernels with one hyperparameter
'''
A = np.array([-1, 1])
h = 0.01
NoN = 2
NoT = 1
NoP = int(round((1-h)/h, 0))
b = np.linspace(h, 1-h, NoP)
guesses = np.random.uniform(low=A[0], high=A[1], size=(NoT, NoN))


variance = np.ones((NoP, NoT))
X_optimal = np.ones((NoP, NoN))
X_nodes = np.ones((NoP, NoT, NoN))

'''
Running the tests
'''
for k in range(0, NoP):
    kernel = kernels.PolynomialBasis(input_dim=1, weight=b[k])
    #kernel = kernels.PolynomialBasisFinite(input_dim=1, weight=b[k])
    for i in range(0, NoT):
        X_nodes[k][i], variance[k][i] = functions.Optimal_Quadrature_Nodes_Optimizer(kernel=kernel, number_of_nodes=NoN, initialguess=guesses, return_var=True)

'''
Finding the minimal variance produced by these tests
'''

for i in range(0, NoP):
    for k in range(0, NoT):
        X_optimal[i] = X_nodes[i][np.argmin(np.absolute(variance[i]))]

'''
Checking if there is already an existing savefile for these parameters, comparing and merging new results to existing ones if possible 
'''
filename = kernel.__class__.__name__ + ', ' + '{}'.format(NoN) + ', ' + '{}'.format(h) + '.txt'

try:
    file = np.loadtxt('./Nodes/' + filename)
    #Merging new results with existing ones
    X_optimal_nodes = np.zeros((len(file), NoN))
    var_1 = np.zeros((len(file), 1))
    var_2 = np.zeros((len(file), 1))
    for i in range(0, NoP):
        kernel = kernels.PolynomialBasis(input_dim=1, weight=b[i])
        var_1[i] = functions.get_var(kernel, X_optimal[i])
        var_2[i] = functions.get_var(kernel, file[i])
        if (var_1[i] > var_2[i]):
            X_optimal_nodes[i] = file[i]
        else:
            X_optimal_nodes[i] = X_optimal[i]

    np.savetxt('./Nodes/' + filename, X_optimal_nodes)
except IOError:
    print("File not accessible")
    np.savetxt('./Nodes/' + filename, X_optimal)