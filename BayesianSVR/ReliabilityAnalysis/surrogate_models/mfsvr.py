from utilities import *
from surrogate_models.svr import LSSVR as _LSSVR, Tune as _STune, SVRInfo as _SVRInfo
from dataclasses import dataclass as _dataclass
from optuna import create_study as _create_study, samplers as _samplers, logging as _logging
from optuna.pruners import MedianPruner as _MedianPruner


"""
1. Import XLF, XHF -> Dataset
2. Check if XHF \in XLF else -> Go to 3
3. Train LF Model
4. Predict XHF with LFM (If not 2, concatenate XHF with XLF) -> Complete LF data set
5. Predict hY with LFM -> Yhl
Concatenate XHF and Yhl 


"""




@_dataclass(frozen=True)
class Parameter:
    C: float
    theta: list
    kernel_type: str


@_dataclass
class MFInfo:
    hfnode: dict
    lfnode: dict

    search_space: dict
    n_fold: int = 5
    n_trial: int = 300

    # To be filled during training
    CVE: float = 0.0
    LFModel: Optional[_LSSVR] = None
    MFModel: Any = None
    Param = None


class MFLSSVR:
    kernels = {'GAUSSIAN': Kernels.k_gaussian,
               'MATERN32': Kernels.k_matern32,
               'MATERN52': Kernels.k_matern52,
               }
    _lbestpar: dict

    def __init__(self, param: Parameter, LFModel: _LSSVR):
        self.lfmodel = LFModel
        self.param = param
        self.trainX_: array = None
        self.Psi_: array = None
        self.dual: array = None
        self.rho: float = 0.0
        self._scaler = MinMaxScaler(clip=True)
        self._kernel = MFLSSVR.kernels['{}'.format(param.kernel_type)]

    def train(self, trainX: array, trainy: array, scale=True, use_lfm=False):
        if use_lfm:
            tmpy = self.lfmodel.predict()

        self._scaler.fit(X=trainX[:, :-1], y=trainy)
        if scale:
            trainX_ = np.hstack([self._scaler.transform(trainX[:, :-1]), trainX[:, -1].reshape(-1, 1)])
        else:
            trainX_ = trainX

        Psi = self._kernel(self.param.theta, trainX_, trainX_)
        self.Psi_ = Psi + np.diag(np.ones(len(Psi)) * (1 / self.param.C))

        A = np.vstack([np.append(0, np.ones([1, len(trainX_)])), np.hstack([np.ones([len(trainX_), 1]), self.Psi_])])
        b = np.append(0, trainy).reshape(-1, 1)
        x_sol = np.linalg.solve(A, b)
        self.rho = x_sol[0].item()
        self.dual = x_sol[1:].ravel()
        self.trainX_ = trainX_

    def predict(self, testX: array, scale=True, path_save=None, use_lfm=False, return_lf=False):
        if use_lfm:
            yhl = self.lfmodel.predict(testX).reshape(-1, 1)
            if scale:
                testX_ = np.hstack([self._scaler.transform(testX), yhl.reshape(-1, 1)])
            else:
                testX_ = np.hstack([testX, yhl.reshape(-1, 1)])
        else:
            if scale:
                testX_ = np.hstack([self._scaler.transform(testX[:, :-1]), testX[:, -1].reshape(-1, 1)])
            else:
                testX_ = testX

        # Edit path_save predicitions
        if path_save is not None:
            # Requires Fast Computation
            if testX.shape[0] >= int(1e5):
                split = 50
            elif testX.shape[0] >= int(1e3):
                split = 100
            else:
                split = 10

            tmppath = r'{}\temp'.format(path_save)
            if exists(tmppath):
                rmtree(tmppath)
            makedirs(tmppath)
            tempX = np.array_split(testX_, split)
            for i in range(len(tempX)):
                K: array = self._kernel(self.param.theta, self.trainX_, tempX[i])
                pred = (np.dot(self.dual.reshape(1, -1), K)).ravel() + float(self.rho)
                np.save(r'{}\{}.npy'.format(tmppath, i), pred)
            del tempX
            temp = []
            for i in range(split):
                temp.append(np.load(r'{}\{}.npy'.format(tmppath, i)))
            pred_ = np.array(temp).ravel()
            rmtree(tmppath)
            return pred_
        else:
            K: array = self._kernel(self.param.theta, self.trainX_, testX_)
            if return_lf:
                return (np.dot(self.dual.reshape(1, -1), K)).ravel() + float(self.rho), testX_[:, -1]
            else:
                return (np.dot(self.dual.reshape(1, -1), K)).ravel() + float(self.rho)


class MFTune:
    _space: dict = {'lb': {'C': 1, 'theta': 0.001, 'eps': 0.0001},
                    'ub': {'C': 100, 'theta': 1, 'eps': 0.01},
                    'kernel_type': 'MATERN32'}

    _mfX: array
    _mfy: array

    _hfnode: dict
    _lfnode: dict
    _best: MFLSSVR

    def __init__(self, info: MFInfo) -> None:
        self.t_info = info
        self.nftr = info.hfnode['trainX'].shape[1]
        MFTune._hfnode = info.hfnode
        MFTune._lfnode = info.lfnode

    def callback(self, study, trial):
        if study.best_trial.number == trial.number:
            self.t_info.MFModel = MFTune._best
            self.t_info.MFModel.train(MFTune._mfX, MFTune._mfy)

    def _cross_validate(self, model: Optional[MFLSSVR]):
        rmse = lambda y1, y2: np.sqrt(np.average((y1 - y2) ** 2))
        kf = KFold(n_splits=self.t_info.n_fold)
        hfX, hfy = MFTune._hfnode['trainX'], MFTune._hfnode['trainy']
        lfX, lfy = MFTune._lfnode['trainX'], MFTune._lfnode['trainy']
        error = []
        try:
            # Get index of hfX samples in lfX array
            coX = np.array([np.where(np.all(lfX == row, axis=1))[0][0] for row in hfX])
            MFTune._mfX, MFTune._mfy = np.hstack([hfX, lfy[coX].reshape(-1, 1)]), hfy
        except IndexError:
            # Use LF model to predict hfX samples
            MFTune._mfX, MFTune._mfy = np.hstack([hfX, self.t_info.LFModel.predict(hfX).reshape(-1, 1)]), hfy

        for tridx, teidx in kf.split(MFTune._mfX, MFTune._mfy):
            model.train(MFTune._mfX[tridx], MFTune._mfy[tridx])
            pred = model.predict(MFTune._mfX[teidx])
            error.append(rmse(MFTune._mfy[teidx], pred))
        cv_error = (1 / self.t_info.n_fold) * sum(error)
        return cv_error

    def objective(self, trial):
        if bool(self.t_info.search_space):
            theta = [trial.suggest_uniform('theta{}'.format(i), self.t_info.search_space['lb']['theta'],
                                           self.t_info.search_space['ub']['theta']) for i in range(self.nftr + 1)]

            C = trial.suggest_loguniform('C', self.t_info.search_space['lb']['C'], self.t_info.search_space['ub']['C'])

            model = MFLSSVR(Parameter(C=C, theta=theta, kernel_type=self.t_info.search_space['kernel_type']), self.t_info.LFModel)

        else:
            theta = [trial.suggest_uniform('theta{}'.format(i), MFTune._space['lb']['theta'],
                                           MFTune._space['ub']['theta']) for i in range(self.nftr + 1)]
            C = trial.suggest_loguniform('C', MFTune._space['lb']['C'], MFTune._space['ub']['C'])
            model = MFLSSVR(Parameter(C=C, theta=theta, kernel_type=MFTune._space['kernel_type']), self.t_info.LFModel)

        cv_error = self._cross_validate(model)
        MFTune._best = model
        return cv_error

    def execute(self, params: Optional[dict] = None, log=True, train_lfm=True):

        if train_lfm:
            lfinfo = _SVRInfo(MFTune._lfnode['trainX'], MFTune._lfnode['trainy'], self.t_info.search_space, 'LSSVR')
            _STune(lfinfo).run()
            self.t_info.LFModel = lfinfo.Model

        """ Optuna Method """
        if not log:
            _logging.set_verbosity(_logging.WARNING)
        study = _create_study(direction='minimize',
                              sampler=_samplers.TPESampler(n_startup_trials=25, n_ei_candidates=100),
                              pruner=_MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10))
        if params is not None:
            study.enqueue_trial(params)
        study.optimize(self.objective, n_trials=self.t_info.n_trial, callbacks=[self.callback])
        self.t_info.CVE = study.best_value
        self.t_info.Param = study.best_params


def camelback(x1, x2, fidelity):
    sc = lambda a, b: (4 * a ** 2) - (2.1 * a ** 4) + (a ** 6) / 3 + (a * b) - (4 * b ** 2) + (4 * b ** 4)
    if fidelity == 'h':
        return sc(x1, x2)
    else:
        return sc(0.7 * x1, 0.7 * x2) + (x1 * x2) + np.sin(x1) + (2 * x2) + np.sin(x2) + x1


if __name__ == "__main__":
    """ Analytical Fxn """
    # xhf = np.random.uniform(low=-2, high=2, size=(10, 2))
    # xlf = np.random.uniform(low=-2, high=2, size=(300, 2))
    # idx = np.random.choice(len(xlf), len(xhf), replace=False)
    # idx.sort()
    # for i in range(len(xhf)):
    #     xlf[idx[i], :] = xhf[i, :]
    #
    # yhf = camelback(xhf[:, 0], xhf[:, 1], 'h')
    # ylf = camelback(xlf[:, 0], xlf[:, 1], 'l')
    #
    # space = {'lb': {'C': 1, 'theta': 0.001},
    #          'ub': {'C': 100, 'theta': 100},
    #          'kernel_type': 'MATERN32'}
    #
    # info = MFInfo(hfnode={'trainX': xhf, 'trainy': yhf}, lfnode={'trainX': xlf, 'trainy': ylf}, search_space=space, n_trial=300)
    # MFTune(info).execute()
    # model = info.MFModel
    # XX, YY = np.meshgrid(np.linspace(-2, 2, num=int(1e3), endpoint=True), np.linspace(-2, 2, num=int(1e3), endpoint=True))
    # # Call LFM predict for XX
    # zlf = camelback(XX.ravel(), YY.ravel(), 'l')
    # Z = model.predict(np.hstack([np.c_[XX.ravel(), YY.ravel()], zlf.reshape(-1, 1)]), scale=True)
    # Z = Z.reshape(XX.shape)
    #
    # fig1 = plt.figure(1)
    # ax1 = fig1.gca(projection='3d')
    # ax1.plot_surface(XX, YY, Z, cmap=plt.get_cmap('nipy_spectral'))

    """ Flutter """

    hf_data = np.loadtxt('..\HF_DATA.dat')
    lf_data = np.loadtxt('..\LF_DATA.dat')
    hf_data = np.loadtxt(r'..\SIMULATIONS\MULTIFIDELITY\1-HF-5.dat')
    lf_data = np.loadtxt(r'..\SIMULATIONS\MULTIFIDELITY\LF-100.dat')

    xhf, yhf = hf_data[:, :-1], hf_data[:, -1]
    xlf, ylf = lf_data[:, :-1], lf_data[:, -1]

    space = {'lb': {'C': 1, 'theta': 0.001},
             'ub': {'C': 100, 'theta': 100},
             'kernel_type': 'GAUSSIAN',
             'ARD': True}

    mfinfo = MFInfo(hfnode={'trainX': xhf, 'trainy': yhf}, lfnode={'trainX': xlf, 'trainy': ylf},
                    search_space=space,
                    n_trial=300)
    MFTune(mfinfo).execute()
    mfmodel: MFLSSVR = mfinfo.MFModel

    XX, YY = np.meshgrid(np.linspace(0.6, 0.905, num=int(1e3), endpoint=True),
                         np.linspace(0.4, 2.0, num=int(1e3), endpoint=True))
    Z = mfmodel.predict(np.c_[XX.ravel(), YY.ravel()], use_lfm=True).reshape(XX.shape)
    fig = plt.figure(1)
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(xhf[:, 0], xhf[:, 1], yhf, s=10, c='r', label='HF')
    ax.scatter(xlf[:, 0], xlf[:, 1], ylf, s=10, c='k', label='LF')
    ax.view_init(azim=101, elev=30)
    ax.plot_surface(XX, YY, Z, cmap=plt.get_cmap('terrain'))

    ax = fig.add_subplot(121)
    contour = ax.contourf(XX, YY, Z, cmap=plt.get_cmap('terrain'), alpha=0.8)
    cbar = fig.colorbar(contour)
    cs = ax.contour(XX, YY, Z, [0.0], colors='k', linewidths=2)
    dat0 = cs.allsegs[0][0]
    ax.plot(dat0[:, 0], dat0[:, 1])
