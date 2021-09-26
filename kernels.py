import GPy
import numpy as np
from GPy.kern.src.kern import Kern
from GPy.core import Param

class PolynomialBasis(Kern):

    def __init__(self, input_dim, weight=0.5, active_dims=None):
        super(PolynomialBasis, self).__init__(input_dim, active_dims, 'poly_basis')
        assert input_dim == 1, "For this kernel we assume input_dim=1"
        self.weight = Param('weight', weight)

    def parameters_changed(self):
        # nothing to do here
        pass

    def K(self, X, X2):
        if X2 is None: X2 = X
        return 1 / (1 - self.weight * X * X2.T)

    def Kdiag(self, X):
        return (1 / (1 - self.weight * X * X)) * np.ones(X.shape[0])

    def update_gradients_full(self, dL_dK, X, X2):
        if X2 is None: X2 = X

        dw = X * X2 / ((-self.weight * X * X2 + 1) ** 2)

        self.weight.gradient = np.sum(dw * dL_dK)

class PolynomialBasisFinite(Kern):

    def __init__(self, input_dim, weight=0.5, active_dims=None):
        super(PolynomialBasisFinite, self).__init__(input_dim, active_dims, 'poly_basis_finite')
        assert input_dim == 1, "For this kernel we assume input_dim=1"
        self.weight = Param('weight', weight)

    def parameters_changed(self):
        # nothing to do here
        pass

    def K(self, X, X2):
        if X2 is None: X2 = X
        n = 8
        upper = (1-np.power((self.weight * X * X2.T), n + 1))
        lower = 1 - self.weight * X * X2.T
        return upper / lower

    def Kdiag(self, X):
        n = 5
        upper = (1 - np.power((self.weight * X * X), n + 1))
        lower = 1 - self.weight * X * X
        return upper / lower

    def update_gradients_full(self, dL_dK, X, X2):
        if X2 is None: X2 = X
        n = 5
        upper1 = - np.power((self.weight * X * X2), n + 1) * (n + 1)
        upper2 = X * X2 * (1 -  np.power((self.weight * X * X2), n + 1))
        lower1 = self.weight * ( - self.weight * X * X2 + 1)
        lower2 =  np.square( - self.weight * X * X2 + 1)
        dw = upper1 / lower1 + upper2 / lower2

        self.weight.gradient = np.sum(dw * dL_dK)

class Trigonometric(Kern):

    def __init__(self, input_dim, weight=0.5, active_dims=None):
        super(Trigonometric, self).__init__(input_dim, active_dims, 'trigonometric')
        assert input_dim == 1, "For this kernel we assume input_dim=1"
        self.weight = Param('weight', weight)

    def parameters_changed(self):
        # nothing to do here
        pass

    def K(self, X, X2):
        if X2 is None: X2 = X
        up = 0.5 * (1 - np.square(self.weight))
        down = 1 + np.square(self.weight) - 2 * self.weight * np.cos(2 * np.pi * (X - X2.T))
        return 0.5 + up / down

    def Kdiag(self, X):
        up = 0.5 * (1 - np.square(self.weight))
        down = 1 + np.square(self.weight) - 2 * self.weight * np.cos(2 * np.pi * (X - X))
        return (0.5 + up/down) * np.ones(X.shape[0])

    def update_gradients_full(self, dL_dK, X, X2):
        if X2 is None: X2 = X

        dw = 1 / np.square(self.weight - 1)

        self.weight.gradient = np.sum(dw * dL_dK)

