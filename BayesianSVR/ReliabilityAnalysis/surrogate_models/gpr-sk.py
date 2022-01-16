from utilities import *

from dataclasses import dataclass as _dataclass
from sklearn.gaussian_process import GaussianProcessRegressor as _gpr
from sklearn.gaussian_process.kernels import Matern as _Matern, RBF as _RBF


@_dataclass
class GPRInfo:
    trainX: array
    trainy: array
    kernel: str
    noise_variance: float = float(1e-5)


class GPR:
    def __init__(self, info: GPRInfo) -> None:
        self.gpmodel: Optional[_gpr] = None
        self._info = info
        self._scaler = MinMaxScaler(clip=True)

        lengthscale = np.array([0.01 for _ in range(info.trainX.shape[1])])
        if info.kernel == 'MATERN32':
            self._kernel = _Matern(length_scale=lengthscale, nu=1.5)
        elif info.kernel == 'MATERN52':
            self._kernel = _Matern(length_scale=lengthscale, nu=2.5)
        else:
            self._kernel = _RBF(length_scale=lengthscale)

    def train(self, scale=True):
        trainX, trainy = self._info.trainX, self._info.trainy
        if scale:
            self._scaler.fit(trainX, trainy)
            trainX = self._scaler.transform(trainX)

        self.gpmodel = _gpr(kernel=self._kernel, alpha=self._info.noise_variance, n_restarts_optimizer=100)
        self.gpmodel.fit(trainX, trainy)

    def predict(self, testX: array, return_std=False, scale=True):
        if scale:
            testX = self._scaler.transform(testX)

        if return_std:
            Z, std = self.gpmodel.predict(testX, return_std=True)
            return Z, std
        else:
            return self.gpmodel.predict(testX, return_std=False)














