from utilities import *
import torch as th
from dataclasses import dataclass as _dataclass
from gpytorch.models import ExactGP as _ExactGP
from gpytorch.means import ConstantMean as _ConstantMean
from gpytorch.kernels import ScaleKernel as _ScaleKernel, RBFKernel as _RBFKernel
from gpytorch.constraints import LessThan as _LessThan, Interval as _Interval
from gpytorch.distributions import MultivariateNormal as _MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood as _GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood as _ExactMarginalLogLikelihood
from gpytorch.settings import fast_pred_var as _fast_pred_var

tarray = Optional[th.Tensor]

"""
If a model is to be reloaded, ensure reload=True, trainX, trainy is given
Any array input into this script should be numpy, arrays converted here
 
"""


@_dataclass(frozen=True)
class GPRInfo:
    path_save: str
    save_model: bool = True
    likelihood: str = 'Gaussian'
    learn_rate: float = 0.1
    train_iter: int = 100
    kernel: str = 'RBF'
    noise_variance: float = float(1e-5)


class BaseGPR(_ExactGP):
    def __init__(self, trainX, trainy, likelihood):
        super(BaseGPR, self).__init__(trainX, trainy, likelihood)
        self.mean_module = None
        self.covar_module = None

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return _MultivariateNormal(mean_x, covar_x)


class GPR:
    base: BaseGPR
    _like: Optional[_GaussianLikelihood]
    _reload: bool = False

    def __init__(self, info: Optional[GPRInfo]):
        self.trainX: array = None
        self.trainy: array = None
        self._info = info
        self._scaler: MinMaxScaler = MinMaxScaler(clip=True)

    def reload(self, model_path):
        state = th.load('{}\Model.back'.format(model_path))
        GPR._like = _GaussianLikelihood()
        trainX_ = self._scaler.transform(self.trainX.copy())
        trainy_ = self.trainy.copy()
        GPR.base = BaseGPR(th.from_numpy(trainX_), th.from_numpy(trainy_), GPR._like)
        GPR.base.mean_module = _ConstantMean()
        GPR.base.covar_module = _ScaleKernel(_RBFKernel(ard_num_dims=trainX_.shape[1]))
        GPR.base.load_state_dict(state['model'])
        GPR._reload = True

    def train(self, trainX: array, trainy: array, scale=True, **param):
        self.trainX = trainX.copy()
        self.trainy = trainy.copy()
        trainy_ = th.from_numpy(trainy).double()
        if scale:
            self._scaler.fit(trainX)
            trainX_ = self._scaler.transform(trainX)
            trainX_ = th.from_numpy(trainX_).double()
        else:
            trainX_ = th.from_numpy(trainX).double()

        # # Initialize Model
        GPR._like = _GaussianLikelihood().double()
        GPR.base = BaseGPR(trainX_, trainy_, GPR._like).double()
        GPR.base.mean_module = _ConstantMean()
        GPR.base.covar_module = _ScaleKernel(_RBFKernel(ard_num_dims=trainX_.shape[1], lengthscale_constraint=_Interval(0.01, 0.1)))
        if 'lengthscale' in param:
            length = param['lengthscale']
        else:
            length = [0.1 for _ in range(trainX_.shape[1])]
        # GPR.base.covar_module.base_kernel.lengthscale = th.tensor(length)
        # GPR.base.likelihood.noise = th.tensor(.01)

        # # Set to training mode
        GPR.base.train()
        GPR._like.train()
        # optimizer = th.optim.Adam(GPR.base.parameters(), lr=self._info.learn_rate)
        optimizer = th.optim.LBFGS(GPR.base.parameters(), lr=self._info.learn_rate, max_iter=100)

        # # Main Training Loop
        # "Loss" for GPs - the marginal log GPR._like
        mll = _ExactMarginalLogLikelihood(GPR._like, GPR.base)

        # # LBFGS Optimizer Loop
        for i in range(self._info.train_iter):
            def closure():
                optimizer.zero_grad()
                output = GPR.base(trainX_)
                loss = -mll(output, trainy_)*trainX_.shape[0]
                loss.backward()
                print('Iter: {} | Loss: {}'.format(i, loss.item()))
                return loss
            optimizer.step(closure)

        # # Adam Optimizer Loop
        # for i in range(self._info.train_iter):
        #     # Zero gradients from previous iteration
        #     optimizer.zero_grad()
        #     # Output from GPR.base
        #     output = GPR.base(trainX_)
        #     # Calc loss and backprop gradients
        #     loss = -mll(output, trainy_)*trainX_.shape[0]
        #     loss.backward()
        #     print('Iter: {} | Loss: {} | Lengthscale: {} | Noise: {}'.format(i, loss.item(),
        #                                                                      GPR.base.covar_module.base_kernel.lengthscale,
        #                                                                      GPR.base.likelihood.noise.item()))
        #     optimizer.step()

        # # Save Model
        if self._info.save_model:
            state = {'model': GPR.base.state_dict(), 'optimizer': optimizer.state_dict()}
            th.save(state, '{}\Model.back'.format(self._info.path_save))

    def predict(self, testX: array, return_var=False, scale=True):
        if scale:
            testX_ = self._scaler.transform(testX)
            testX_ = th.from_numpy(testX_)
        else:
            testX_ = testX

        GPR._like.eval()
        GPR.base.eval()

        fpreds = GPR.base(testX_)
        if return_var:
            return fpreds.mean.detach().numpy(), fpreds.variance.detach().numpy()
        else:
            return fpreds.mean.detach().numpy()

        # # Set to predictive mode
        # with th.no_grad(), _fast_pred_var():
        #     observed_pred = GPR._like(GPR.base(testX_.double()))
        #     if return_std:
        #         return observed_pred.mean.numpy(), observed_pred.stddev.numpy()
        #     else:
        #         return observed_pred.mean.numpy()


