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

#Search interval of specified b-values
b_low = 0.45
b_high = 0.5
h = 0.01

if (b_low == h and b_high == 1-h):
    NoP = int(round((1 - h) / h, 0))
    index1 = 0
    index2 = NoP
    b = np.linspace(h, 1 - h, NoP)
else:
    NoP = int(round((b_high-h)/h, 0)) - int(round((b_low-h)/h, 0)) + 1
    index1 = int(b_low * (1/h) - 1)
    index2 = int(b_high * (1/h) - 1)
    b = np.linspace(b_low, b_high, NoP)

A = np.array([-1, 1])
h = 0.01
NoN = 6
NoT = 10
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
        X_nodes[k][i], variance[k][i] = functions.Optimal_Quadrature_Nodes_Optimizer(kernel=kernel, number_of_nodes=NoN, initialguess=guesses[i], return_var=True)

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
        var_2[i] = functions.get_var(kernel, file[index1 + i])
        if (var_1[i] > var_2[i]):
            file[index1 + i] = file[index1 + i]
        else:
            file[index1 + i] = X_optimal[i]

    np.savetxt('./Nodes/' + filename, file)
except IOError:
    print("File not accessible")
    if (b_low == h and b_high == 1 - h):
        np.savetxt('./Nodes/' + filename, X_optimal)
    else:
        print("Please run the process for all b-values first, not specific ones.")