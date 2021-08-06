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