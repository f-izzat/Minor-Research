from typing import List, Tuple

import numpy as np
from gpflow.likelihoods import Gaussian
from emukit.examples.multi_fidelity_dgp.multi_fidelity_deep_gp import DGP_Base, init_layers_mf

class MFDGP:

    def __init__(self, X: List, Y: List, kernels: List, Z=None, mean_function=None, whiten=False):
        self._Y = Y
        self._X = X

        if Z is None:
            self.Z = self._make_inducing_points(X, Y)
        else:
            self.Z = Z

        layers = init_layers_mf(Y, self.Z, kernels, num_outputs=Y[0].shape[1], mean_function=mean_function, whiten=whiten)
        self.n_layers = len(layers)
        likelihood = Gaussian(variance=1e-8)
        likelihood.trainable = False

        self.model = DGP_Base(X, Y, likelihood, layers)
        self.n_fidelities = len(X)

    @staticmethod
    def _make_inducing_points(X: List, Y: List) -> List:
        """
        Makes inducing points at every other training point location which is deafult behaviour if no inducing point
        locations are passed

        :param X: training locations
        :param Y: training targets
        :return: List of numpy arrays containing inducing point locations
        """
        Z = [X[0].copy()]
        for x, y in zip(X[:-1], Y[:-1]):
            Z.append(np.concatenate((x.copy()[::2], y.copy()[::2]), axis=1))
        return Z

    def optimize(self, iterations: list, lr=None, multi_step=False, fix_inducing=True):
        """
        Optimize variational parameters alongside kernel and likelihood parameters using the following regime:
            1. Optimize the parameters while fixing the intermediate layer mean variational parameters
            2. Free the mean of the variational distribution and optimize all parameters together
        """

        lr = [0.01, 0.003, 0.003] if lr is None else lr
        if fix_inducing:
            self.model.fix_inducing_point_locations()

        if multi_step:
            print('Multi step Optimization')
            print('-'*100)
            self.model.layers[0].q_mu = self._Y[0]
            for i, layer in enumerate(self.model.layers[1:-1]):
                layer.q_mu = self._Y[i][::2]
                layer.q_mu.trainable = False

            self.model.fix_inducing_point_locations()
            self.model.multi_step_training(lr_1=lr[0], lr_2=lr[1], n_iter=iterations[0], n_iter_2=iterations[1])

            for layer in self.model.layers[:-1]:
                layer.q_mu.trainable = True
            print('-' * 100)
            print('Final Optimization: {} Iterations'.format(iterations[2]))
            print('-' * 100)
            self.model.run_adam(lr[2], iterations[2])

        else:
            self.model.fix_inducing_point_locations()
            self.model.layers[0].q_mu = self._Y[0]
            for i, layer in enumerate(self.model.layers[1:-1]):
                layer.q_mu = self._Y[i][::2]
                layer.q_mu.trainable = False

            for ii in range(self.n_layers):
                if ii == 0:
                    self.model.layers[0].q_sqrt = self.model.layers[0].q_sqrt.value * 1e-8
                    self.model.layers[0].q_sqrt.trainable = False
                self.model.likelihood.likelihood.variance = self._Y[-1].var() * 1e-8
                self.model.likelihood.likelihood.variance.trainable = False
            print('-' * 100)
            print('Optimization 1: {} Iterations'.format(iterations[0]))
            print('-' * 100)
            self.model.run_adam(lr[0], iterations[0])
            # self.model.layers[0].q_sqrt.trainable = True
            print('-' * 100)
            print('Optimization 2: {} Iterations'.format(iterations[1]))
            print('-' * 100)
            self.model.run_adam(lr[1], iterations[1])

            for layer in self.model.layers[:-1]:
                layer.q_mu.trainable = True

            print('-' * 100)
            print('Optimization 3: {} Iterations'.format(iterations[2]))
            print('-' * 100)
            self.model.run_adam(lr[2], iterations[2])


    def predict(self, X: np.array, predict_f=False) -> Tuple[np.array, np.array]:
        y_m, y_v = self.model.predict_y(X, 250)
        y_mean_high = np.mean(y_m, axis=0).flatten()
        y_var_high = np.mean(y_v, axis=0).flatten() + np.var(y_m, axis=0).flatten()

        if not predict_f:
            return y_mean_high[:, None], y_var_high[:, None]
        else:
            self.model.predict_f(X, S=1)
