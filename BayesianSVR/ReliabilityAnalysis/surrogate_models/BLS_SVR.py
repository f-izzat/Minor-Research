from utilities import *
from optuna import create_study as _create_study, samplers as _samplers, logging as _logging
from optuna.pruners import MedianPruner as _MedianPruner

tmp_path = 'D:\PYTHON\.PROJECTS\BayesianSVR\SIMULATIONS\.temp'

"""
    Test structure, where only class Front is accessed
"""


@dataclass(frozen=True)
class _Parameter:
    C: float
    theta: list
    ktype: str = 'GAUSSIAN'


class _BLSSVR:
    kernels = {'GAUSSIAN': Kernels.k_gaussian,
               'MATERN32': Kernels.k_matern32,
               'MATERN52': Kernels.k_matern52,
               }

    def __init__(self, param: _Parameter):
        self.param = param
        self.trainX_ = None
        self.dual: array = None
        self.rho: float = 0.0
        self.loglike: float = 0.0

        self._kernel = _BLSSVR.kernels['{}'.format(param.ktype)]
        self.Psi_ = None
        self._scaler = StandardScaler()

    def train(self, trainX: array, trainy: array, likelihood=True):
        self._scaler.fit(X=trainX, y=trainy)
        trainX_ = self._scaler.transform(trainX)
        Psi = self._kernel(self.param.theta, trainX_, trainX_)
        self.Psi_ = Psi + np.diag(np.ones(len(Psi)) * (1 / self.param.C))

        A = np.vstack([np.append(0, np.ones([1, len(trainX_)])), np.hstack([np.ones([len(trainX_), 1]), self.Psi_])])
        b = np.append(0, trainy).reshape(-1, 1)
        x_sol = np.linalg.solve(A, b)
        self.rho = x_sol[0].item()
        self.dual = x_sol[1:].ravel()
        self.trainX_ = trainX_

        if likelihood:
            fmp = self.predict(trainX_.copy(), scale=False)
            loss = 0.5 * (trainy - fmp) ** 2
            zd = 0.5 * (len(trainX) * np.log(2 * self.param.C) + np.log(np.linalg.det(Psi + np.diag(np.ones(len(Psi)) * self.param.C))))
            t1 = self.param.C * np.sum(loss)
            t2 = 0.5 * (self.dual.reshape(1, -1) @ Psi @ self.dual.reshape(-1, 1))
            t3 = len(trainX) * np.log(np.sqrt(2 * np.pi / self.param.C))
            self.loglike = (t1 + t2 + t3 + zd).item()

    def predict(self, testX: array, scale=True, return_var=False):

        if scale:
            testX_ = self._scaler.transform(testX)
        else:
            testX_ = testX

        if testX_.shape[0] >= int(1e5):
            split = 5000
            tempX = np.array_split(testX_, split)
            mean = np.array_split(np.zeros(testX_.shape[0], ), split)
            if return_var:
                Psi_ = np.linalg.inv(self.Psi_)
                var = np.array_split(np.zeros(testX_.shape[0], ), split)
                for i in range(len(tempX)):
                    tmpKtr = self._kernel(self.param.theta, tempX[i], self.trainX_)
                    tmpKte = np.diag(self._kernel(self.param.theta, tempX[i], tempX[i])).ravel()
                    var[i] = tmpKte - np.diag((tmpKtr @ Psi_ @ tmpKtr.T)).ravel()
                    mean[i] = (tmpKtr @ self.dual.reshape(-1, 1)).ravel() + self.rho
                    del tmpKtr, tmpKte
                return np.array(mean).ravel(), np.array(var).ravel()
            else:
                for i in range(len(tempX)):
                    tmpKtr = self._kernel(self.param.theta, tempX[i], self.trainX_)
                    mean[i] = (tmpKtr @ self.dual.reshape(-1, 1)).ravel() + self.rho
                    del tmpKtr
        else:
            Ktr = self._kernel(self.param.theta, testX_, self.trainX_)
            mean = (Ktr @ self.dual.reshape(-1, 1)).ravel() + self.rho
            if return_var:
                Psi_ = np.linalg.inv(self.Psi_)
                Kte = np.diag(self._kernel(self.param.theta, testX_, testX_)).ravel()
                return mean, Kte - np.diag((Ktr @ Psi_ @ Ktr.T)).ravel()
            else:
                return mean


class Front:
    _space: dict = {'lb': {'C': 0.1, 'theta': 0.001},
                    'ub': {'C': 1, 'theta': 1},
                    'kernel_type': 'GAUSSIAN'}
    _trainX: array
    _trainy: array
    _best: Optional[_BLSSVR]
    _n_trial: int = 100

    def __init__(self, trainX: array, trainy: array, method: str, **kwargs):
        # Method - 'single' -> input parameter
        # Method - 'tune' -> search space
        Front._trainX = trainX
        Front._trainy = trainy
        self.model: Optional[_BLSSVR] = None
        self.params: dict = {}

        if method == 'single':
            assert 'C' in kwargs, "Parameter C is not given"
            assert 'theta' in kwargs, "Parameter theta is not given"
            assert len(kwargs['theta']) == trainX.shape[1], "Incompatible theta dimension"
            self._single(kwargs)

        if method == 'tune':
            self._tune(kwargs)

    @staticmethod
    def _logging(log) -> None:
        if not log:
            _logging.set_verbosity(_logging.WARNING)
        else:
            return None

    @staticmethod
    def _objective(trial):
        C = trial.suggest_uniform('C', Front._space['lb']['C'], Front._space['ub']['C'])
        theta = [trial.suggest_uniform('theta{}'.format(i), Front._space['lb']['theta'], Front._space['ub']['theta'])
                 for i in range(Front._trainX.shape[1])]
        # for i in range(Front._trainX.shape[1]):
        #     theta.append(trial.suggest_uniform('theta{}'.format(i), Front._space['lb']['theta'], Front._space['ub']['theta']))

        params = _Parameter(C=C, theta=theta, ktype=Front._space['kernel_type'])
        _model = _BLSSVR(params)
        _model.train(Front._trainX.copy(), Front._trainy.copy(), likelihood=True)
        Front._best = _model
        return -_model.loglike

    def _callback(self, study, trial):
        if study.best_trial.number == trial.number:
            self.model = Front._best

    def _single(self, args: dict):
        if 'kernel' in args:
            param = _Parameter(C=args['C'], theta=args['theta'], ktype=args['kernel'])
        else:
            param = _Parameter(C=args['C'], theta=args['theta'])
        self.model = _BLSSVR(param)
        if 'likelihood' in args:
            self.model.train(Front._trainX.copy(), Front._trainy.copy(), likelihood=args['likelihood'])
        else:
            self.model.train(Front._trainX.copy(), Front._trainy.copy())

    def _tune(self, args: dict):
        study = _create_study(direction='minimize',
                              sampler=_samplers.TPESampler(n_startup_trials=25),
                              pruner=_MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10))
        if 'search_space' in args:
            Front._space = args['search_space']

        if 'log' in args:
            if args['log']:
                pass
            else:
                self._logging(args['log'])

        if 'n_trial' in args:
            Front._n_trial = args['n_trial']

        if 'params' in args:
            study.enqueue_trial(args['params'])
        study.optimize(lambda trial: self._objective(trial), n_trials=Front._n_trial,
                       callbacks=[self._callback])
        self.params = study.best_params


if __name__ == "__main__":
    data = np.loadtxt('D:\PYTHON\.PROJECTS\BayesianSVR\SIMULATIONS\.REFERENCE\.REF_ESVR-N\Data.dat')
    model = Front(trainX=data[:, :-1], trainy=data[:, -1], method='tune', n_trial=50)
