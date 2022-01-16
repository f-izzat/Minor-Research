import sys
from multiprocessing import Process
from os import makedirs
from typing import Union
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import tensorflow as tf
from gpflow import set_trainable
from gpflux.models.deep_gp import DeepGP
from gpflux.optimization.keras_natgrad import NatGradWrapper
from gpflow.optimizers import NaturalGradient
from scipy.stats import norm, normaltest, kstest, skew, kurtosis
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from reliability_base import Problem, DGPConfig, LayersConfig
sys.path.insert(0, "E:/23620029-Faiz/.PROJECTS/AdaptiveDGP")
tf.keras.backend.set_floatx("float64")
tf.get_logger().setLevel("INFO")
np.set_printoptions(threshold=sys.maxsize, precision=4, suppress=True)

rng = np.random.RandomState(123)

def four_branch(XX: np.ndarray, k: int = 7) -> np.ndarray:
    assert XX.ndim == 2, 'X.ndim != 2'
    X1, X2 = XX[:, 0], XX[:, 1]
    tmp1 = 3 + (0.1*(X1 - X2)**2) - ((X1 + X2)/np.sqrt(2))
    tmp2 = 3 + (0.1*(X1 - X2)**2) + ((X1 + X2)/np.sqrt(2))
    tmp3 = (X1 - X2) + k/np.sqrt(2)
    tmp4 = -(X1 - X2) + k/np.sqrt(2)
    return np.min((tmp1, tmp2, tmp3, tmp4), axis=0).reshape(-1, 1)


class AdaptiveDGP:

    __slots__ = '_X', '_Y', '_problem', '_modelcfg', '_layerscfg', 'model'

    def __init__(self, problem: Problem, model_cfg: DGPConfig) -> None:
        self._problem = problem
        self._modelcfg = model_cfg
        self._layerscfg = model_cfg.layers_config

        self._X, self._Y = problem.data
        likelihood_layer, hidden_layers = self._layerscfg.likelihood_layer, self._layerscfg.hidden_layers
        self.model = DeepGP(f_layers=hidden_layers, likelihood=likelihood_layer)

    def optimize(self):
        train_mode = NatGradWrapper(self.model.as_training_model())
        train_mode.compile([NaturalGradient(gamma=gg) for gg in self._layerscfg.gamma] +
                           [tf.optimizers.Adam(self._modelcfg.learning_rate)])
        train_mode.fit({"inputs": self._X, "targets": self._Y},
                       epochs=int(self._modelcfg.epoch),
                       verbose=self._modelcfg.verbose)

    def update(self, XX: np.ndarray, YY: np.ndarray):
        XX = XX.reshape(-1, 1) if XX.ndim < 2 else XX
        YY = YY.reshape(-1, 1) if YY.ndim < 2 else YY
        self._X = np.vstack([self._X, XX])
        self._Y = np.vstack([self._Y, YY])
        self._layerscfg.X = self._X.copy()
        self._layerscfg.update_layers()
        likelihood_layer, hidden_layers = self._layerscfg.likelihood_layer, self._layerscfg.hidden_layers
        self.model = DeepGP(f_layers=hidden_layers, likelihood=likelihood_layer)
        self.optimize()

    def predict(self, XX: np.ndarray, return_std: bool = False):
        model = self.model.as_prediction_model()
        pred = model(XX)

        if return_std:
            return pred.f_mean.numpy(), np.sqrt(pred.f_var.numpy())
        else:
            return pred.f_mean.numpy()

    def sample_posterior(self, layer_input, _iter, n_samples= int(1e4)):
        _path = f'{self._problem.main_path}/STATISTICS'
        # stats = np.stack([stats_block for _ in range(len(self.model.f_layers))], axis=-1)
        count = [ii for ii in range(len(self.model.f_layers))]
        for layer, cc in zip(self.model.f_layers, count):
            if cc == 0:
                stats_block = np.concatenate([layer_input.copy(), np.zeros([len(layer_input), 4])], axis=-1)
            else:
                stats_block = np.concatenate([layer_input.numpy(), np.zeros([len(layer_input), 4])], axis=-1)
            layer.num_samples = n_samples
            layer_output = layer(layer_input)
            samples = tf.convert_to_tensor(layer_output) # 3D Array (num_samples, len(layer_input), 1)
            layer_input = samples[0] # 2D Array (len(layer_input), 1)
            temp = samples[:, :, 0].numpy() # 2D Array (num_samples, len(layer_input)) -> take along rows axis=0
            stats_block[:, 2] = skew(temp, axis=0)
            stats_block[:, 3] = kurtosis(temp, axis=0)
            # stats_block[:, 4:6, cc] = np.array([kstest(temp[:, ii], 'norm', alternative='greater') for ii in range(temp.shape[1])])
            # stats_block[:, 6:8, cc] = np.array([kstest(temp[:, ii], 'norm', alternative='less') for ii in range(temp.shape[1])])
            stats_block[:, 4:6] = np.array(normaltest(temp, axis=0)).T # Return p values only
            np.savetxt(f'{_path}/LAYER_{cc}_{_iter}.dat', stats_block)
            layer.num_samples = None


class Reliability:

    __slots__ = '_problem', 'iteration', 'adaptive_model'
    def __init__(self, problem: Problem, model_cfg: DGPConfig) -> None:
        
        if type(model_cfg) == DGPConfig:
            self.adaptive_model = AdaptiveDGP(problem=problem, model_cfg=model_cfg)
        else:
            NotImplementedError()
        
        self._problem = problem
        self.iteration = 0
        # # --------- Create Files ----------#
        self._problem.log_path = f'{self._problem.main_path}/LOG'
        self._problem.plot_path = f'{self._problem.main_path}/PLOTS'
        self._problem.surface_path = f'{self._problem.plot_path}/SURFACE'
        self._problem.stats_path = f'{self._problem.main_path}/STATISTICS'
        makedirs(self._problem.log_path)
        makedirs(self._problem.plot_path)
        makedirs(self._problem.surface_path)
        makedirs(self._problem.stats_path)
        self._log(file_name='initial')
        self.adaptive_model.optimize()

    @staticmethod
    def _gen_mcs(shape: tuple) -> np.ndarray:
        n_ftr = shape[1]
        n_mcs = int(1e6)
        samples = np.random.normal(loc=0.0, scale=1.0, size=(n_mcs, n_ftr))
        return samples

    @staticmethod
    def _u_fxn(mu: np.ndarray, std: np.ndarray):
        val = np.abs(mu.ravel()) / std.ravel()
        idx = np.argmin(val)
        return idx

    @staticmethod
    def _ef_fxn(mu: np.ndarray, std: np.ndarray, a=0):
        eps = 2 * std**2
        t1 = (a - mu) / std
        t2 = (a - eps - mu) / std
        t3 = (a + eps - mu) / std
        eff = (mu - a) * (2 * norm.cdf(t1) - norm.cdf(t2) - norm.cdf(t3)) \
              - std * (2 * norm.pdf(t1) - norm.pdf(t2) - norm.pdf(t3)) \
              + (norm.cdf(t3) - norm.cdf(t2))
        idx = np.argmax(eff)
        return idx

    @staticmethod
    def _er_fxn(mu: np.ndarray, std:np.ndarray):
        mu, std = mu.ravel(), std.ravel()
        sign = np.sign(mu).ravel()
        term1 = (-sign * mu) * norm.cdf(-sign * (mu / std))
        term2 = std * norm.pdf(mu / std)
        erf = term1 + term2
        idx = np.argmax(erf)
        return idx

    def _log(self, file_name: str, **kwargs):

        file_name = file_name.upper()
        if file_name == 'INITIAL':
            # # Adaptive
            _file = open(f'{self._problem.log_path}/ADAPTIVE.dat', 'w')
            _file.write('Iteration\t POF\t COV\t P_PLUS\t P_MIN\t Pf\n')
            _file.close()
            # # Added Samples
            _file = open(f'{self._problem.log_path}/SAMPLES.dat', 'w')
            _file.close()
            # # Surrogate
            _file = open(f'{self._problem.log_path}/SURROGATE.dat', 'w')
            _file.write('Iteration\t RMSE\n')
            _file.close()
        elif file_name == 'ADAPTIVE':
            args = ['pof', 'cov', 'p_plus', 'p_minus']
            assert not set(args) - set(list(kwargs.keys()))
            _file = open(f'{self._problem.log_path}/ADAPTIVE.dat', 'a')
            _file.write(f"{self.iteration}\t {kwargs['pof']}\t {kwargs['cov']}\t {kwargs['p_plus']}\t"
                        f" {kwargs['p_minus']}\t {(kwargs['p_plus'] - kwargs['p_minus']) / kwargs['pof']}\n")
            _file.close()
        elif file_name == 'SAMPLES':
            args = ['X', 'Y']
            assert not set(args) - set(list(kwargs.keys()))
            assert kwargs['X'].ndim == 2 and kwargs['Y'].ndim == 2
            _file = open(f'{self._problem.log_path}/SAMPLES.dat', 'ab')
            np.savetxt(_file, np.hstack([kwargs['X'], kwargs['Y']]))
            _file.close()
        elif file_name == 'SURROGATE':
            assert 'rmse' in kwargs
            _file = open(f'{self._problem.log_path}/SURROGATE.dat', 'a')
            _file.write(f"{self.iteration}\t {kwargs['rmse']}\n")
            _file.close()
        else:
            return NotImplementedError 
    
    def _plot(self, mc_samp, new_samp: bool = True):
        plt.ioff()
        plt_rng = self._problem.plot_range
        surrg_rng = self._problem.surrogate_range
        fig = plt.figure(1, figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.set_xlim(plt_rng[0])
        ax.set_ylim(plt_rng[1])

        lines, line_labels = [], []
        points, point_labels = [], []
        # plt_obj, plt_labels = [], []

        XX, YY = np.meshgrid(np.linspace(plt_rng[0][0], plt_rng[0][1], 100),
                             np.linspace(plt_rng[1][0], plt_rng[1][1], 100))

        WW = np.c_[XX.ravel(), YY.ravel()]        
        # # --------- True Model --------- #
        ZZ = self._problem.test_fxn(WW).reshape(XX.shape)
        true_cs = ax.contour(XX, YY, ZZ, [0.0], colors='k', linewidths=1, zorder=1.0)
        # #
        lines.append(true_cs.collections[0])
        line_labels.append('True LSF')

        # plt_obj.append(true_cs.collections[0])
        # plt_labels.append('True LSF')
        # #

        # # --------- Surrogate --------- #
        XX, YY = np.meshgrid(np.linspace(surrg_rng[0][0], surrg_rng[0][1], 100),
                             np.linspace(surrg_rng[1][0], surrg_rng[1][1], 100))

        WW = np.c_[XX.ravel(), YY.ravel()]
        mu = self.adaptive_model.predict(WW)

        # Surface
        fig_surf = go.Figure()
        fig_surf.add_trace(go.Surface(x=XX, y=YY, z=mu.reshape(XX.shape), opacity=0.9, colorscale='Viridis'))
        fig_surf.add_trace(go.Scatter3d(name='Initial Samples',
                                        x=self._problem.data[0][:, 0],
                                        y=self._problem.data[0][:, 1],
                                        z=self._problem.data[1].ravel(),
                                        mode='markers', marker=dict(size=3, symbol="square", color="magenta")))

        # Contour
        model_cs = ax.contour(XX, YY, mu.reshape(XX.shape), [0.0], colors='b', linewidths=1, linestyles='dashed', zorder=1.0)
        contour = self.adaptive_model.predict(true_cs.allsegs[0][0])
        rmse = mean_squared_error(np.zeros(contour.shape), contour, squared=True)
        _file = open(f'{self._problem.log_path}/SURROGATE.dat', 'a')
        _file.write(f'{self.iteration}\t {rmse}\n')
        _file.close()

        # # TODO: CHANGE BACK TO  plt_labels.append(f'{self._probelm.name} LSF')
        lines.append(model_cs.collections[0])
        line_labels.append('DGP LSF')


        # # --------- Contour Plot Labels --------- #
        initial_points = ax.scatter(self._problem.data[0][:, 0], self._problem.data[0][:, 1],
                                    c='m', marker='s', s=20, alpha=0.5, zorder=2)
        points.append(initial_points)
        point_labels.append('Initial Samples')


        mcs_pred, mcs_std = self.adaptive_model.predict(mc_samp, return_std=True)
        mcs_pred = mcs_pred.ravel()
        mcs_std = mcs_std.ravel()

        mcs_mu_plot = plt.figure(2)
        mcs_mu_plot = plt.hist(mcs_pred, histtype='stepfilled', bins=int(len(mcs_pred)/10))
        mcs_mu_plot = plt.gcf().savefig(f'{self._problem.stats_path}/{self.iteration}-MU.png', dpi=100)
        mcs_mu_plot = plt.clf()

        mcs_std_plot = plt.figure(3)
        mcs_std_plot = plt.hist(mcs_std, histtype='stepfilled', bins=int(len(mcs_std)/10))
        mcs_std_plot = plt.gcf().savefig(f'{self._problem.stats_path}/{self.iteration}-STD.png', dpi=100)
        mcs_std_plot = plt.clf()


        pos, neg = np.argwhere(mcs_pred > 0).ravel(), np.argwhere(mcs_pred < 0).ravel()

        mcs_pos = ax.scatter(mc_samp[pos, 0], mc_samp[pos, 1], c='springgreen', marker='o', edgecolor='seagreen', s=10, alpha=0.5, zorder=0.1)

        points.append(mcs_pos)
        point_labels.append('Predicted Safe Region')


        mcs_neg = ax.scatter(mc_samp[neg, 0], mc_samp[neg, 1], c='lightgray', marker='o', edgecolor='dimgray', s=10, alpha=0.5, zorder=0.1)

        points.append(mcs_neg)
        point_labels.append('Predicted Fail Region')

        if new_samp:
            added_samples = np.loadtxt(f"{self._problem.log_path}/SAMPLES.dat")
            if added_samples.ndim < 2:
                added_samples = added_samples.reshape(1, -1)
            added_points = ax.scatter(added_samples[:, 0], added_samples[:, 1],
                                      c='red', marker='x', s=15, alpha=1.0, zorder=3)

            points.append(added_points)
            point_labels.append('Added Samples')
            fig_surf.add_trace(go.Scatter3d(name='Added Samples',
                                            x=added_samples[:, 0],
                                            y=added_samples[:, 1],
                                            z=added_samples[:, -1],
                                            mode='markers', marker=dict(size=3, symbol="x", color="red")))

        points_legend = plt.legend(points, point_labels, loc='upper center', bbox_to_anchor=(0.5, 1.11), shadow=False,
                                   ncol=len(point_labels), fontsize='small', markerscale=2)

        lines_legend = plt.legend(lines, line_labels, loc='upper left', ncol=1, shadow=False,
                                  fontsize='small', markerscale=2)
        plt.gca().add_artist(points_legend)
        plt.gca().add_artist(lines_legend)
        fig_surf.update_layout(autosize=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        fig_surf.write_html(f'{self._problem.surface_path}/{self.iteration}.html', auto_open=False)
        fig.savefig(f'{self._problem.plot_path}/{self.iteration}.png', dpi=100)
        plt.close('all')      

    def _learn(self, mu: np.ndarray, std: np.ndarray, k=2):
        # transformer = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=123)
        # mu = transformer.fit_transform(mu)
        # std = transformer.fit_transform(std)
        if self._problem.learn_fxn.lower() == 'erf':
            idx = self._er_fxn(mu=mu, std=std)
        elif self._problem.learn_fxn.lower() == 'eff':
            idx = self._ef_fxn(mu=mu, std=std)
        else:
            idx = self._u_fxn(mu, std)

        p_plus = np.sum((mu.ravel()-k*std.ravel()) <= 0) / len(mu)
        p_minus = np.sum((mu.ravel()+k*std.ravel()) <= 0) / len(mu)
        pof = np.sum(mu.ravel() <= 0) / len(mu)
        p_f = (p_plus- p_minus) / pof
        cov = np.sqrt((1 - pof) / (pof * len(mu)))
        self._log(file_name='adaptive', pof=pof, cov=cov, p_plus=p_plus, p_minus=p_minus)
        return idx, pof, cov, p_f

    def execute(self, n_updates: int):
        mcs_shape = (self._problem.n_mcs, self._problem.data[0].shape[1])
        samples = self._gen_mcs(shape=mcs_shape)
        self._plot(new_samp=False, mc_samp=samples)
        for self.iteration in range(1, n_updates + 1):
            mu, std = self.adaptive_model.predict(samples, return_std=True)
            # stats_test = samples[np.random.choice(len(samples), size=500, replace=False), :]
            # self.adaptive_model.sample_posterior(stats_test, self.iteration)
            idx, pof, cov, p_f = self._learn(mu, std)
            new_sample = np.hstack([samples[idx, :].reshape(1, -1), self._problem.test_fxn(samples[idx, :].reshape(1, -1))])
            self._log(file_name='samples', X=new_sample[:, :-1].reshape(1, -1), Y=new_sample[:, -1].reshape(1, -1))
            self.adaptive_model.update(new_sample[:, :-1], new_sample[:, -1])
            print(f'{self._problem.model_name}: Update {self.iteration}| Sample = {new_sample} | CoV = {cov:.6f} | Pf = {p_f:.6f}')
            self._plot(new_samp=True, mc_samp=samples)
            # samples = np.delete(samples, idx, axis=0)
            if p_f <= 0.01:
                print('-'*50, f'{self._problem.model_name} CONVERGED ', '-'*50)
                break
            if cov > 0.03:
                samples = self._gen_mcs(shape=mcs_shape)

        # #
        _log = np.genfromtxt(f'{self._problem.log_path}/ADAPTIVE.dat')[1:, :]
        iterations, _pof, _cov, _pplus, _pmin, _pf = _log[:, 0], _log[:, 1], _log[:, 2], _log[:, 3], _log[:, 4], _log[:, 5]
        fig = plt.figure(figsize=(8, 6))
        plt.plot(iterations, _pof, c='k')
        plt.xlabel('Iterations')
        plt.xlim([iterations[0], iterations[-1]])
        plt.xticks(np.arange(0, 110, step=10))
        plt.ylabel(r'$P_{f}$')
        fig.savefig(f'{self._problem.plot_path}/POF.png', dpi=100)
        plt.close('all')

        fig = plt.figure(figsize=(8, 6))
        plt.plot(iterations, _cov, c='k')
        plt.xlabel('Iterations')
        plt.xlim([iterations[0], iterations[-1]])
        plt.xticks(np.arange(0, 110, step=10))
        plt.ylabel('Coefficient of Variation')
        fig.savefig(f'{self._problem.plot_path}/COV.png', dpi=100)
        plt.close('all')

        fig = plt.figure(figsize=(8, 6))
        plt.plot(iterations, _pmin, c='k', label=r'$P^{+}_{f}$', linewidth=1)
        plt.plot(iterations, _pplus, c='r', label=r'$P^{-}_{f}$', linewidth=1)
        plt.legend()
        plt.xticks(np.arange(0, 110, step=10))
        plt.xlabel('Iterations')
        plt.xlim([iterations[0], iterations[-1]])
        plt.ylabel(r'$P_{f}$')
        fig.savefig(f'{self._problem.plot_path}/PP.png', dpi=100)
        plt.close('all')

        fig = plt.figure(figsize=(8, 6))
        plt.plot(iterations, _pf, c='k', linewidth=1)
        plt.xlabel('Iterations')
        plt.xticks(np.arange(0, 110, step=10))
        plt.xlim([iterations[0], iterations[-1]])
        plt.ylabel(r'$\frac{P^{+}_{f} - P^{-}_{f}}{P^{0}_{f}}$''')
        fig.savefig(f'{self._problem.plot_path}/Pf.png', dpi=100)
        plt.close('all')
        # #


def execute_block(block, _exp_name, _exp_number):
    exec(block % (_exp_name, _exp_number))


def define_experiment(_name, _consistency, _processes) -> list:
    _path = f'E:/23620029-Faiz/.PROJECTS/AdaptiveDGP/Reliability/Simulations/GPFLUX/DGP/INITIAL_1/EXPERIMENTS/{_name[1:-1]}.txt'
    _file = open(_path, 'r')
    _block = _file.read()
    _file.close()
    for nn in _consistency:
        p = Process(target=execute_block, args=(_block, _name, nn))
        p.start()
        _processes.append(p)
    return _processes


if __name__ == "__main__":

    processes = []
    # exp_name = "'1L-1'"
    # exp_numbers = [6]
    # processes = define_experiment(_name=exp_name, _consistency=exp_numbers, _processes=processes)
    #
    # exp_name = "'3L-1'"
    # exp_numbers = [1]
    # processes = define_experiment(_name=exp_name, _consistency=exp_numbers, _processes=processes)
    #
    exp_name = "'2L-1'"
    exp_numbers = [7]
    processes = define_experiment(_name=exp_name, _consistency=exp_numbers, _processes=processes)

    for proc in processes:
        proc.join()











