import sys
import gpflow
import numpy as np
import tensorflow as tf
from gpflow import set_trainable
from gpflow.kernels import SquaredExponential
from gpflux.helpers import (
    construct_basic_inducing_variables,
    construct_basic_kernel,
    construct_mean_function,
)
from gpflux.layers.gp_layer import GPLayer
from gpflow.likelihoods import Gaussian
from dataclasses import dataclass
from typing import Any, Union, List
sys.path.insert(0, "E:/23620029-Faiz/.PROJECTS/AdaptiveDGP")
tf.keras.backend.set_floatx("float64")
tf.get_logger().setLevel("INFO")

@dataclass
class Problem:    
    data: tuple
    test_fxn: Any
    model_name: str
    learn_fxn: str    
    main_path: str
    surrogate_range: list
    plot_range: list
    cov_req: float = 0.05
    stats_path: Union[None, str] = None
    plot_path: Union[None, str] = None
    surface_path: Union[None, str] = None
    log_path: Union[None, str] = None
    n_mcs: int = 1e6

@dataclass
class LayersConfig:
    X: np.ndarray    
    input_dims: list
    output_dims: list
    gamma: list
    whiten: bool = True
    lengthscale: float = 2.0
    q_sqrt_factor: float = 1e-5
    likelihood_variance: float = 1e-5
    train_likelihood: bool = True
    num_latent_gps: Union[None, int] = None
    hidden_layers: Union[None, List[GPLayer]] = None
    likelihood_layer: Union[None, Gaussian] = None
    num_samples: Union[None, int] = None

    def update_layers(self):
        self.hidden_layers = []
        _X = self.X.copy()
        # _Z = np.linspace(tuple(self.X.min(0)), tuple(self.X.max(0)), len(self.X))
        _Z = _X.copy()

        in_dim = self.input_dims
        out_dim = self.output_dims

        for ii in range(len(in_dim)):
            is_last_layer = ii == len(in_dim) - 1

            inducing = construct_basic_inducing_variables(num_inducing=_Z.shape[1],
                                                          input_dim=in_dim[ii],
                                                          share_variables=True, z_init=_Z)
            if is_last_layer:
                mean_function = gpflow.mean_functions.Zero()
                q_sqrt_scaling = 1.0
                kern = SquaredExponential(variance=1.0, lengthscales=[self.lengthscale] * in_dim[ii])
            else:
                mean_function = construct_mean_function(_X, in_dim[ii], out_dim[ii])
                _X = mean_function(_X)
                if tf.is_tensor(_X):
                    _X = _X.numpy()
                q_sqrt_scaling = self.q_sqrt_factor
                kern = SquaredExponential(variance=1e-6, lengthscales=[2.] * in_dim[ii])

            kernel = construct_basic_kernel(kern, output_dim = out_dim[ii], share_hyperparams = True)
            layer = GPLayer(kernel, inducing, len(_X), mean_function=mean_function,
                            name=f"HIDDEN_{ii}", num_latent_gps=self.num_latent_gps, num_samples=self.num_samples)
            layer.q_sqrt.assign(layer.q_sqrt * q_sqrt_scaling)
            self.hidden_layers.append(layer)

    def __post_init__(self):
        # Perform Checks here
        assert len(self.input_dims) == len(self.output_dims)        
        assert len(self.gamma) == len(self.output_dims)        
        #
        self.update_layers()
        likelihood = Gaussian(variance=self.likelihood_variance)
        if not self.train_likelihood:
            set_trainable(likelihood.variance, False)
        self.likelihood_layer = likelihood

# @dataclass
# class LayersConfig2:

@dataclass
class DGPConfig:
    layers_config: LayersConfig
    num_samples: int =100
    learning_rate: float = 0.01
    epoch: Union[list, int] = 5000
    verbose: int = 0
    multi_step: bool = False




