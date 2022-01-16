from utilities import *
from gpflow.models import GPR as _GPR
from gpflow import kernels as _kernels
from gpflow.optimizers import Scipy as _Scipy
from gpflow.utilities import read_values as _read_values, multiple_assign as _multiple_assign, set_trainable as _set_trainable
from gpflow.config import set_default_float
from gpflow.config import default_float
from tensorflow import convert_to_tensor
from tensorflow import random as _tfrand
from dataclasses import dataclass as _dataclass

set_default_float(np.float64)
np.random.seed(0)
_tfrand.set_seed(0)


@_dataclass(frozen=True)
class GPRInfo:
    path_save: str
    save_model: bool = True  # If true, ModelParam saved in path_save
    def_kernel: str = 'RBF'
    def_noise: float = float(1e-5)


class GPR:
    _opt: _Scipy
    _model: _GPR

    def __init__(self, info: Optional[GPRInfo]):
        self.trainX: array = None
        self.trainy: array = None

        self._info = info
        self._scaler: MinMaxScaler = MinMaxScaler(clip=True)

    def reload(self):
        ModelParam = jload(r'{}\Model.back'.format(self._info.path_save))
        trainX_ = convert_to_tensor(self._scaler.transform(self.trainX.copy()), dtype=default_float())
        trainy_ = convert_to_tensor(self.trainy.reshape(-1, 1), dtype=default_float())
        length = [0.1 for _ in range(self.trainX.shape[1])]
        GPR._model = _GPR(data=(trainX_, trainy_), kernel=_kernels.RBF(lengthscales=length))
        _multiple_assign(GPR._model, ModelParam)

    def train(self, trainX: array, trainy: array, scale=True):
        self.trainX = trainX.copy()
        self.trainy = trainy.reshape(-1, 1).copy()
        if scale:
            self._scaler.fit(trainX)
            trainX_ = self._scaler.transform(trainX)
            trainX_ = convert_to_tensor(trainX_, dtype=default_float())
        else:
            trainX_ = convert_to_tensor(trainX)
        trainy_ = convert_to_tensor(trainy.reshape(-1, 1), dtype=default_float())
        length = [0.1 for _ in range(trainX.shape[1])]
        # _kernel = _kernels.RBF(lengthscales=length) + _kernels.Constant()
        _kernel = _kernels.RBF(lengthscales=length)
        GPR._model = _GPR(data=(trainX_, trainy_), kernel=_kernel, noise_variance=float(1e-5))
        _set_trainable(GPR._model.likelihood.variance, False)

        GPR._opt = _Scipy()
        GPR._opt.minimize(GPR._model.training_loss, GPR._model.trainable_variables, options=dict(maxiter=1000))
        if self._info.save_model:
            ModelParam = _read_values(GPR._model)
            jdump(ModelParam, r'{}\Model.back'.format(self._info.path_save))

    def predict(self, testX: array, return_var=False, scale=True):
        if scale:
            testX_ = self._scaler.transform(testX)
        else:
            testX_ = convert_to_tensor(testX, dtype=default_float())

        mean, var = GPR._model.predict_f(testX_)
        if return_var:
            return mean.numpy().ravel(), var.numpy().ravel()
        else:
            return mean.numpy().ravel()




