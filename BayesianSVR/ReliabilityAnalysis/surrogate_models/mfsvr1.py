from utilities import *
from surrogate_models.svr import LSSVR as _LSSVR, Tune as _STune, SVRInfo as _SVRInfo
from dataclasses import dataclass as _dataclass
from optuna import create_study as _create_study, samplers as _samplers, logging as _logging, pruners as _pruners, \
    trial as _trial

"""
1. Import XLF, XHF -> Dataset
2. Check if XHF \in XLF else -> Go to 3
3. Train LF Model
4. Predict XHF with LFM (If not 2, concatenate XHF with XLF) -> Complete LF data set
5. Predict hY with LFM -> Yhl
Concatenate XHF and Yhl 
Ensure X data is 2D

"""


@_dataclass(frozen=True)
class Parameter:
    C: float
    theta: list
    kernel_type: str
    lossfxn: int = 2


class MFInfo:
    search_space: dict
    subset: Union[bool, list] = False
    n_fold: int = 5
    mf_trial = 300
    lf_trial = 300

    def __init__(self, hfX: array, hfy: array, lfX: array, lfy: array, search_space: dict,
                 n_fold=5, mf_trial=300, **kwargs):
        self.hfX = hfX
        self.hfy = hfy
        self.lfX = lfX
        self.lfy = lfy

        MFInfo.search_space = search_space
        MFInfo.n_fold = n_fold
        MFInfo.mf_trial = mf_trial

        if 'lf_trial' in kwargs:
            MFInfo.lf_trial = kwargs['lf_trial']

        self.LFModel: Union[_LSSVR, None] = None
        self._trainLF()

        self.mfX = np.hstack([self.hfX, self.LFModel.predict(self.hfX).reshape(-1, 1)])
        self.mfy = hfy

        self._checksubset()

        # Tune Results
        self.MFModel = None
        self.CVE: float = 0.0
        self.Param: dict = {}

    def _checksubset(self) -> None:
        """ Check if XHF \in XLF by iteration over rows """
        ih, il = np.where((self.hfX == self.lfX[:, None]).all(-1))
        if len(ih) == 0:
            self.lfX = np.vstack([self.lfX, self.hfX])
            self.lfy = np.append(self.lfy, self.LFModel.predict(self.hfX))

    def _trainLF(self):
        lfinfo = _SVRInfo(self.lfX, self.lfy, MFInfo.search_space, n_trial=MFInfo.lf_trial, svrtype='LSSVR')
        _STune(lfinfo).run()
        self.LFModel = lfinfo.Model


class MFLSSVR:
    kernels = {'GAUSSIAN': Kernels.k_gaussian,
               'MATERN32': Kernels.k_matern32,
               'MATERN52': Kernels.k_matern52,
               }

    def __init__(self, param: Parameter, LFModel: _LSSVR, scaler: str = 'MIN-MAX'):
        self._kernel = MFLSSVR.kernels['{}'.format(param.kernel_type)]

        self.param = param
        self.lfmodel = LFModel
        self.scaler = Scaler(method=scaler)
        self.trainX_: array = None
        self.Psi_: array = None
        self.dual: array = None
        self.rho: float = 0.0

    def train(self, trainX: array, trainy: array, scale=True):
        if trainX.shape[1] != len(self.param.theta):
            trainX = np.hstack([trainX, self.lfmodel.predict(trainX).reshape(-1, 1)])

        # Train X -> mfX = [xhf yhl]
        self.scaler.fit(trainX[:, :-1])
        if scale:
            trainX_ = np.hstack([self.scaler.scale(trainX[:, :-1]), trainX[:, -1].reshape(-1, 1)])
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

    def predict(self, testX: array, scale=True):
        if testX.shape[1] == self.trainX_.shape[1]-1:
            testX = np.hstack([testX, self.lfmodel.predict(testX).reshape(-1, 1)])
        if scale:
            testX_ = np.hstack([self.scaler.scale(testX[:, :-1]), testX[:, -1].reshape(-1, 1)])
        else:
            testX_ = testX
        K: array = self._kernel(self.param.theta, self.trainX_, testX_)
        return (np.dot(self.dual.reshape(1, -1), K)).ravel() + float(self.rho)


class MFTune:
    _best: MFLSSVR

    def __init__(self, info: MFInfo):
        self.info = info

    def _callback(self, study, trial):
        if study.best_trial.number == trial.number:
            self.info.MFModel = MFTune._best
            self.info.MFModel.train(self.info.mfX, self.info.mfy)

    def _crossval(self, model: MFLSSVR):
        rmse = lambda y1, y2: np.sqrt(np.average((y1 - y2) ** 2))
        kf = KFold(n_splits=self.info.n_fold)
        error = []

        for tridx, teidx in kf.split(self.info.mfX, self.info.mfy):
            model.train(self.info.mfX[tridx], self.info.mfy[tridx])
            pred = model.predict(self.info.mfX[teidx])
            error.append(rmse(self.info.mfy[teidx], pred))
        cv_error = (1 / self.info.n_fold) * sum(error)
        return cv_error

    def _objective(self, trial):
        theta = [trial.suggest_uniform('theta{}'.format(i), self.info.search_space['lb']['theta'],
                                       self.info.search_space['ub']['theta']) for i in range(self.info.mfX.shape[1])]

        C = trial.suggest_loguniform('C', self.info.search_space['lb']['C'], self.info.search_space['ub']['C'])

        model = MFLSSVR(Parameter(C=C, theta=theta, kernel_type=self.info.search_space['kernel_type']),
                        self.info.LFModel)
        cv_error = self._crossval(model)
        MFTune._best = model
        return cv_error

    def optimize(self, log=True, x0: Union[dict, None] = None):
        if not log:
            _logging.set_verbosity(_logging.WARNING)
        study = _create_study(direction='minimize',
                              sampler=_samplers.TPESampler(n_startup_trials=25, n_ei_candidates=100),
                              pruner=_pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10))
        if x0 is not None:
            study.enqueue_trial(x0)
        study.optimize(self._objective, n_trials=self.info.mf_trial, callbacks=[self._callback])
        self.info.CVE = study.best_value
        self.info.Param = study.best_params


# if __name__ == "__main__":
#     hf_data = np.loadtxt(r'..\__SIMULATIONS\MULTIFIDELITY\1-HF\1-HF-25.dat')
#     lf_data = np.loadtxt(r'..\__SIMULATIONS\MULTIFIDELITY\LF-100.dat')
#
#     hfX, hfy = hf_data[:, :-1], hf_data[:, -1]
#     lfX, lfy = lf_data[:, :-1], lf_data[:, -1]
#
#     space = {'lb': {'C': 1, 'theta': 0.001},
#              'ub': {'C': 100, 'theta': 100},
#              'kernel_type': 'GAUSSIAN',
#              'ARD': True}
#
#     mfinfo = MFInfo(hfX=hfX, hfy=hfy, lfX=lfX, lfy=lfy, search_space=space, mf_trial=100, lf_trial=200)
#     MFTune(mfinfo).optimize()
#     mfmodel: MFLSSVR = mfinfo.MFModel
#
#     XX, YY = np.meshgrid(np.linspace(0.6, 0.905, num=int(1e3), endpoint=True),
#                          np.linspace(0.4, 2.0, num=int(1e3), endpoint=True))
#     Z = mfmodel.predict(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
#     fig = plt.figure(1)
#     ax = fig.add_subplot(122, projection='3d')
#     ax.scatter(hfX[:, 0], hfX[:, 1], hfy, s=10, c='r', label='HF')
#     ax.scatter(lfX[:, 0], lfX[:, 1], lfy, s=10, c='k', label='LF')
#     ax.view_init(azim=101, elev=30)
#     ax.plot_surface(XX, YY, Z, cmap=plt.get_cmap('terrain'))
#
#     ax = fig.add_subplot(121)
#     contour = ax.contourf(XX, YY, Z, cmap=plt.get_cmap('terrain'), alpha=0.8)
#     cbar = fig.colorbar(contour)
#     cs = ax.contour(XX, YY, Z, [0.0], colors='k', linewidths=2)
#     dat0 = cs.allsegs[0][0]
#     ax.plot(dat0[:, 0], dat0[:, 1])



