import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from gpflow.kernels import RBF, White, Linear, Matern32, Matern52
from gpflow.mean_functions import Identity
from mfdgp import MFDGP
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import iqr
from emukit.examples.multi_fidelity_dgp.baseline_model_wrappers import LinearAutoRegressiveModel, NonLinearAutoRegressiveModel
from idaes.surrogate.pysmo.sampling import LatinHypercubeSampling as LHS

"""
Notes
- Add whitened representation, although OBJ_with_white > OBJ_without_white 
- Best 20-60-4
- 20-60-5 added ELBO, KL and MNLL info (stopped early) -> reload, try change S=10 in build_predict, propagate and DGP_Base __init__ and num_samples=250 for prediction
    - Stopped early again, 20-60-6 better
- 20-60-6 try num_samples=250 for prediction (Monte Carlo Samples)
- 20-120-4 try num_samples=250 for prediction in MFDGP, 20-60-5 settings reverted
"""

folder = '20-120-5'
nhf = 20
nlf = 120

def make_kernel(X, Y, add_linear=False):
    n_fidelities = len(X)
    Din = X[0].shape[1]
    Dout = Y[0].shape[1]

    _kernels = [RBF(Din, active_dims=list(range(Din)), variance=1., lengthscales=1., ARD=True)]
    for l in range(1, n_fidelities):
        D = Din + Dout
        D_range = list(range(D))
        k_corr = RBF(Din, active_dims=D_range[:Din], variance=1., lengthscales=1., ARD=True)
        k_prev = RBF(Dout, active_dims=D_range[Din:], variance=1, lengthscales=1., ARD=True)
        k_in = RBF(Din, active_dims=D_range[:Din], variance=1., lengthscales=1., ARD=True)

        if add_linear:
            k_hf = k_corr * (k_prev + Linear(Dout, active_dims=D_range[Din:], variance=1.)) + k_in
        else:
            k_hf = (k_corr * k_prev) + k_in

        _kernels.append(k_hf)

    """
    A White noise kernel is currently expected by Mf-DGP at all layers except the last.
    In cases where no noise is desired, this should be set to 0 and fixed, as follows:

        white = White(1, variance=0.)
        white.variance.trainable = False
        _kernels[i] += white
    """
    for i, kernel in enumerate(_kernels[:-1]):
        white = White(1, variance=0.)
        white.variance.trainable = False
        _kernels[i] += white

    return _kernels

def find_shared_rows(X):
    # Find shared rows between current high fidelity with 300 main
    Xhf_main = np.loadtxt('{}/FLUTTER/HF_300.dat'.format(path))[:, :-1]
    # Returns a rows in Xhf_main which is in X
    return (Xhf_main[:, None] == X).all(-1).any(-1)

def plot_mf(_model, hf, lf, _XX, _YY, model_type, save_path):

    if model_type == 'MF-DGP':
        Z, var = _model.predict(scaler.fit_transform(np.c_[_XX.ravel(), _YY.ravel()]))
        var = var.ravel()
        Z = Z.reshape(_XX.shape)
    else:
        Z, var = _model.predict(np.hstack([scaler.fit_transform(np.c_[_XX.ravel(), _YY.ravel()]), np.ones([len(_XX.ravel()), 1])]))
        Z = Z.ravel().reshape(_XX.shape)
        var = var.ravel()
    np.savetxt('{}/{}-VAR.dat'.format(save_path, model_type), var)
    fig1 = plt.figure(1)
    ax1 = fig1.gca(projection='3d')
    _surface = ax1.plot_surface(_XX, _YY, Z, cmap=plt.get_cmap('nipy_spectral'))
    ax1.scatter(hf[:, 0], hf[:, 1], hf[:, 2], s=30, marker='x', c='k', label='$n_{{HF}}$ = {}'.format(len(hf)))
    ax1.scatter(lf[:, 0], lf[:, 1], lf[:, 2], s=10, marker='^', c='r', label='$n_{{LF}}$ = {}'.format((len(lf))))
    ax1.set_xlabel('Mach')
    ax1.set_ylabel('Flutter Speed Index')
    ax1.set_zlabel('Damping Coefficient')
    ax1.legend()
    # ax1.set_title(model_type.upper())
    ax1.view_init(azim=120)
    fig1.savefig('{}/{}-Surface.png'.format(save_path, model_type))

    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    _contour = ax2.contourf(_XX, _YY, Z, cmap=plt.get_cmap('nipy_spectral'), alpha=0.7)
    _cbar = fig2.colorbar(_contour)
    _cbar.ax.set_ylabel('Damping Coefficient')
    _cs = ax2.contour(_XX, _YY, Z, [0.0], colors='k', linewidths=0)
    # _dat0 = _cs.allsegs[0][0]
    tmp = _cs.allsegs[0]
    if len(tmp) > 1:
        first = np.vstack(tmp[:-1])
        last = tmp[-1]
        first = first[first[:, 0].argsort()]
        last = last[last[:, 0].argsort()]
        plt.plot(first[:, 0], first[:, 1], '-.', c='k')
        plt.plot(last[:, 0], last[:, 1], '-.', c='k')
        tmp = np.vstack([first, np.array([[np.nan, np.nan]])])
        tmp = np.vstack([tmp, last])
        np.savetxt('{}/{}-FB.dat'.format(save_path, model_type), tmp)
    else:
        _dat0 = np.vstack(_cs.allsegs[0])
        np.savetxt('{}/{}-FB.dat'.format(save_path, model_type), _dat0)
        plt.plot(_dat0[:, 0], _dat0[:, 1], '-.', c='k')

    ax2.scatter(hf[:, 0], hf[:, 1], s=30, marker='x', c='k', label='$n_{{HF}}$ = {}'.format(len(hf)))
    ax2.scatter(lf[:, 0], lf[:, 1], s=10, marker='^', c='r', label='$n_{{LF}}$ = {}'.format((len(lf))))
    ax2.set_xlim(0.6, 0.91)
    ax2.set_ylim(0.4, 2.0)
    ax2.set_xlabel('Mach')
    ax2.set_ylabel('Flutter Speed Index')
    # ax2.set_title(model_type.upper())

    # Shrink current axis's height by 10% on the bottom
    _box = ax2.get_position()
    ax2.set_position([_box.x0, _box.y0 + _box.height * 0.1,
                     _box.width, _box.height * 0.9])
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11), shadow=False, ncol=5)
    fig2.savefig('{}/{}-Contour.png'.format(save_path, model_type))
    plt.close(fig='all')

def run_mfdgp(X, Y, _test, _hf, _lf, experiment, save_fig):
    # # Multifidelity Deep Gaussian Process
    _kernels = make_kernel(X=X, Y=Y, add_linear=True)
    _lf_kernel, _hf_kernel = _kernels[0], _kernels[1]
    # _mean = [Identity(input_dim=(20, 2)) for _ in range(len(X))]
    _model = MFDGP(X=X, Y=Y, kernels=[_lf_kernel, _hf_kernel], mean_function=None, whiten=False)
    _model.optimize(iterations=[7500, 10000, 15000], lr=[1e-3, 1e-6, 1e-5], multi_step=True)
    # _model.optimize(iterations=[10, 10, 15], lr=[1e-3, 1e-6, 1e-5], multi_step=True)
    _ZZ = _model.predict(scaler.fit_transform(_test[:, :-1]))[0].ravel()
    _rmse = mean_squared_error(_test[:, -1], _ZZ, squared=False)
    _yiqr = iqr(_test[:, -1])
    file = open('E:/23620029-Faiz/.PROJECTS/EMUKIT_MFDGP/data/FLUTTER/{}/LOG.dat'.format(folder), "a")
    file.write('EXPERIMENT {} \t MFDGP \t RMSE \t {} \t NORM_RMSE \t {} \t R2 {} \t MNLL \t {} \t ELBO \t {} \t KL \t {} \n'.format(experiment, _rmse, _rmse/_yiqr, r2_score(_test[:, -1], _ZZ),
                                                                                                                                    -_model.model.compute_log_likelihood(),
                                                                                                                                    _model.model.L.eval(session=_model.model.enquire_session()),
                                                                                                                                    _model.model.KL.eval(session=_model.model.enquire_session())))
    _model.model.enquire_session()
    file.close()
    _XX, _YY = np.meshgrid(np.linspace(0.6, 0.91, 100), np.linspace(0.4, 2.0, 100))
    plot_mf(_model, _hf, _lf, _XX, _YY, model_type='MF-DGP', save_path=save_fig)

def run_nargp(X, Y, _test, _hf, _lf, experiment, save_fig):
    _model = NonLinearAutoRegressiveModel(X=X, Y=Y)
    _model.optimize()
    _ZZ = _model.predict(np.hstack([scaler.fit_transform(_test[:, :-1]), np.ones([len(_test), 1])]))[0].ravel()
    _rmse = mean_squared_error(_test[:, -1], _ZZ, squared=False)
    _yiqr = iqr(_test[:, -1])
    file = open('E:/23620029-Faiz/.PROJECTS/EMUKIT_MFDGP/data/FLUTTER/{}/LOG.dat'.format(folder), "a")
    file.write('EXPERIMENT {} \t NARGP \t RMSE \t {} \t NORM_RMSE \t {} \t R2 \t {} \t MNLL \t {} \n'.format(experiment, _rmse, _rmse/_yiqr, r2_score(_test[:, -1], _ZZ), _model.mnll))
    file.close()
    _XX, _YY = np.meshgrid(np.linspace(0.6, 0.91, 100), np.linspace(0.4, 2.0, 100))
    plot_mf(_model, _hf, _lf, _XX, _YY, model_type='NARGP', save_path=save_fig)

def run_argp(X, Y, _test, _hf, _lf, experiment, save_fig):
    _model = LinearAutoRegressiveModel(X=X, Y=Y)
    _model.optimize()
    _ZZ = _model.predict(np.hstack([scaler.fit_transform(_test[:, :-1]), np.ones([len(_test), 1])]))[0].ravel()
    _rmse = mean_squared_error(_test[:, -1], _ZZ, squared=False)
    _yiqr = iqr(_test[:, -1])
    file = open('E:/23620029-Faiz/.PROJECTS/EMUKIT_MFDGP/data/FLUTTER/{}/LOG.dat'.format(folder), "a")
    file.write('EXPERIMENT {} \t AR1 \t RMSE \t {} \t NORM_RMSE \t {} \t R2 \t {} \t MNLL \t {} \n'.format(experiment, _rmse, _rmse/_yiqr, r2_score(_test[:, -1], _ZZ), _model.mnll))
    file.close()
    _XX, _YY = np.meshgrid(np.linspace(0.6, 0.91, 100), np.linspace(0.4, 2.0, 100))
    plot_mf(_model, _hf, _lf, _XX, _YY, model_type='AR1', save_path=save_fig)

if __name__ == "__main__":

    o_hf = np.loadtxt('E:/23620029-Faiz/.PROJECTS/EMUKIT_MFDGP/data/FLUTTER/HF_300.dat')
    o_lf = np.loadtxt('E:/23620029-Faiz/.PROJECTS/EMUKIT_MFDGP/data/FLUTTER/LF_300.dat')

    procs = []

    for ii in range(20):
        exp = ii + 1
        path = 'E:/23620029-Faiz/.PROJECTS/EMUKIT_MFDGP/data/FLUTTER/{}/{}'.format(folder, exp)
        os.makedirs(path)
        print('-' * 100)
        print('EXPERIMENT {}'.format(exp))
        print('-'*100)

        # # Make HF and LF data from LHS
        hf_sampling = LHS(o_hf, nhf, sampling_type='selection')
        lf_sampling = LHS(o_lf, nlf+50, sampling_type='selection')

        hf_data = hf_sampling.sample_points()
        lf_data = lf_sampling.sample_points()

        while not np.logical_and(len(hf_data) == nhf, len(lf_data) == nlf):
            hf_data = hf_sampling.sample_points()
            lf_data = lf_sampling.sample_points()

        shared_idx = (np.isclose(o_hf[:, None], hf_data)).all(-1).any(-1)
        test_ = o_hf[~shared_idx, :]

        # hf_idx = np.random.choice(o_hf.shape[0], nhf, replace=False)
        # lf_idx = np.random.choice(o_lf.shape[0], nlf, replace=False)
        # hf_data = o_hf[hf_idx, :]
        # lf_data = o_lf[lf_idx, :]
        # test_ = np.delete(o_hf, hf_idx, axis=0)
        np.savetxt('{}/HF.dat'.format(path), hf_data)
        np.savetxt('{}/LF.dat'.format(path), lf_data)
        np.savetxt('{}/TEST.dat'.format(path), test_)

        Xhf, yhf = hf_data[:, :-1], hf_data[:, -1].reshape(-1, 1)
        Xlf, ylf = lf_data[:, :-1], lf_data[:, -1].reshape(-1, 1)

        scaler = MinMaxScaler()
        Xhf_sc = scaler.fit_transform(Xhf)
        Xlf_sc = scaler.fit_transform(Xlf)


        run_mfdgp(X=[Xlf_sc, Xhf_sc], Y=[ylf, yhf], _hf=hf_data, _lf=lf_data, _test=test_,
                  experiment=exp, save_fig=path)
        run_nargp(X=[Xlf_sc, Xhf_sc], Y=[ylf, yhf], _hf=hf_data, _lf=lf_data, _test=test_,
                  experiment=exp, save_fig=path)
        run_argp(X=[Xlf_sc, Xhf_sc], Y=[ylf, yhf], _hf=hf_data, _lf=lf_data, _test=test_,
                  experiment=exp, save_fig=path)

        del hf_sampling, lf_sampling




