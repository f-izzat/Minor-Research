from utilities import *

from dataclasses import dataclass as _dataclass
from subprocess import check_call as _check_call, DEVNULL as _DEVNULL
from scipy.special import erf as _erf


from optuna import create_study as _create_study, samplers as _samplers, logging as _logging
from optuna.exceptions import OptunaError as _OptunaError
from optuna.pruners import MedianPruner as _MedianPruner
OPTUNE_EARLY_STOPING = 100


@_dataclass(frozen=True)
class Parameter:
    C: float
    theta: list
    beta: float
    eps: float
    ktype: str


class EarlyStoppingExceeded(_OptunaError):
    early_stop = OPTUNE_EARLY_STOPING
    early_stop_count = 0
    best_score = None


class Tune:
    def __init__(self, ModelInfo) -> None:
        self.model_info = ModelInfo
        self.train_data: array = None
        self._best = None
        self.best_model = None

    @staticmethod
    def early_stopping_opt(study, trial):
        if EarlyStoppingExceeded.best_score is None:
            EarlyStoppingExceeded.best_score = study.best_value

        if study.best_value < EarlyStoppingExceeded.best_score:
            EarlyStoppingExceeded.best_score = study.best_value
            EarlyStoppingExceeded.early_stop_count = 0
        else:
            if EarlyStoppingExceeded.early_stop_count > EarlyStoppingExceeded.early_stop:
                EarlyStoppingExceeded.early_stop_count = 0
                best_score = None
                raise EarlyStoppingExceeded()
            else:
                EarlyStoppingExceeded.early_stop_count = EarlyStoppingExceeded.early_stop_count + 1
        print(
            f'EarlyStop counter: {EarlyStoppingExceeded.early_stop_count}, Best score: {study.best_value} and {EarlyStoppingExceeded.best_score}')
        return

    def callback(self, study, trial):
        if study.best_trial.number == trial.number:
            self.best_model = self._best

    def objective(self, trial, train_data):
        # Normalize train data first
        trainX, trainy = train_data[:, :-1], train_data[:, 1]
        tempX = np.zeros(trainX.shape)
        for i in range(trainX.shape[1]):
            mean = trainX[:, i].mean()
            std = trainX[:, i].std()
            tempX[:, i] = (trainX[:, i] - mean) / std
        tempy = (trainy - trainy.mean()) / trainy.std()
        lbt = [0.1 * abs(max(tempX[:, 0])), 0.1 * abs(max(tempX[:, 1]))]
        ubt = [10 * lbt[0], 10 * lbt[1]]
        search_space = {'lb': {'C': 1 * abs(max(tempy)), 'Epsilon': 0.001 * abs(min(tempy)), 'Beta': float(0.1),
                               'Kaio': float(1e0), 'Kappa_b': float(1e0), 'ARD': 0.1},
                        'ub': {'C': 10 * abs(max(tempy)), 'Epsilon': 1, 'Beta': float(1.0),
                               'Kaio': float(10), 'Kappa_b': float(10), 'ARD': 1}}
        # 1 * abs(min(tempy))
        # C = trial.suggest_uniform('C', search_space['lb']['C'], search_space['ub']['C'])
        # eps = trial.suggest_uniform('eps', search_space['lb']['Epsilon'], search_space['ub']['Epsilon'])
        beta = trial.suggest_uniform('beta', search_space['lb']['Beta'], search_space['ub']['Beta'])
        # beta = 0.9
        # kaio = trial.suggest_uniform('kaio', search_space['lb']['Kaio'], search_space['ub']['Kaio'])
        # kappa_b = trial.suggest_uniform('kappa_b', search_space['lb']['Kappa_b'], search_space['ub']['Kappa_b'])
        # ard = trial.suggest_uniform('ard', search_space['lb']['ARD'], search_space['ub']['ARD'])
        # param = {'C': C, 'Epsilon': eps, 'Beta': beta, 'Kaio': kaio, 'Kappa_b': kappa_b, 'ARD': ard}
        # param = {'C': C, 'Epsilon': eps, 'Beta': beta, 'ARD': ard}
        param = {'Beta': beta}
        model = SILFSVR(self.model_info)
        model.train(train_data, ard=True, params=param)
        self._best = model
        return model.evidence

    def run(self, train_data, n_trial=None, log=True):
        if not log:
            _logging.set_verbosity(_logging.WARNING)
        study = _create_study(direction='minimize',
                              sampler=_samplers.TPESampler(n_startup_trials=25),
                              pruner=_MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10))
        try:
            study.optimize(lambda trial: self.objective(trial, train_data), n_trials=n_trial, callbacks=[self.callback, self.early_stopping_opt])
        except EarlyStoppingExceeded:
            print(f'EarlyStopping Exceeded: No new best scores on iters {OPTUNE_EARLY_STOPING}')
        # study.optimize(lambda trial: self.objective(trial, train_data), n_trials=n_trial, callbacks=[self.callback])


class SILFSVR:
    def __init__(self, ModelInfo: dict) -> None:
        self.path_save: str = ModelInfo['root_path']
        self.path_exe: str = ModelInfo['path_exe']
        self.init_beta: float = ModelInfo['init_beta']
        self.kernel = Kernels.k_silf

        if not exists(self.path_save):
            makedirs(self.path_save)

        # Stored after training
        self.ard: bool = True
        self.params: dict = {}
        self.results: dict = {}
        self.svm_list: dict = {}
        self.mean, self.std = [], []
        self.trainX: array = None
        self.trainy: array = None
        self.trainX_: array = None  # Normalized
        self.alpha: array = None
        self.ndim: int = 0
        self.evidence: float = 0.0

        # Variables to calculate variance
        self.MtrainX: array = None
        self.MPsi: array = None
        self.alpha_off: array = None

    def _loss_fxn(self):
        # Split Sets Cm, Cp, Mm, Mp, Z
        # dual = alpha - alpha*
        if len(self.sv_list['idx_none']) != 0:
            alpha_none = self.alpha[self.sv_list['idx_none']]
        if len(self.sv_list['idx_off']) != 0:
            alpha_off = self.alpha[self.sv_list['idx_off']]
        if len(self.sv_list['idx_on']) != 0:
            alpha_on = self.alpha[self.sv_list['idx_on']]
        pass

    def _likelihood(self):
        if self.ndim == 1:
            KL = [self.params["ARD parameter of dimension 1"]]
        elif self.ndim == 2:
            KL = [self.params["ARD parameter of dimension 1"], self.params["ARD parameter of dimension 2"]]
        theta = [self.params["Kaio"], KL, self.params["Kappa_b"]]
        Psi_ = self.kernel(theta, self.trainX_, self.trainX_)
        fmp = (Psi_ @ self.alpha.reshape(1, -1)).ravel()
        y_ = self.trainy * self.std[-1] + self.mean[-1]
        term1 = 0.5 * (self.alpha.reshape(1, -1) @ Psi_ @ self.alpha.reshape(-1, 1)).ravel()

    def _variance(self, theta, testX):
        print("Computing Variance...")
        C, eps, beta = self.params["C"], self.params["Epsilon"], self.params["Beta"]
        Zs = (2 * eps * (1 - beta)) + (2 * np.sqrt((np.pi * beta * eps) / C).item() * _erf(np.sqrt(C * beta * eps))) + (
                (2 / C) * np.exp(-C * beta * eps).item())
        t1 = ((eps ** 3) * (1 - beta) ** 3) / 3
        t2 = (np.sqrt((np.pi * beta * eps) / C).item()) * ((2 * beta * eps / C) + (eps ** 2) * (1 - beta) ** 2) * _erf(
            np.sqrt(C * beta * eps)).item()
        t3 = (4 * beta * eps ** 2) * (1 - beta) / C
        t4 = ((((eps ** 2) * (1 - beta) ** 2) / C) + ((2 * eps * (1 + beta)) / C ** 2) + (2 / C ** 3)) * np.exp(
            -C * beta * eps).item()
        sigma_n = (2 / Zs) * (t1 + t2 + t3 + t4)

        KMM = self.kernel(theta, self.MtrainX, self.MtrainX)
        diag = np.diag_indices(len(KMM))
        KMM[diag] = KMM[diag] + (2 * beta * eps / C)
        KMM = np.linalg.inv(KMM)

        # Requires Fast Computation
        if testX.shape[0] >= int(1e5):
            split = 5000
        elif testX.shape[0] >= int(1e3):
            split = 100
        else:
            split = 1

        tmppath = r'{}\temp'.format(self.path_save)
        if exists(tmppath):
            rmtree(tmppath)

        makedirs(tmppath)
        tempX = np.array_split(testX, split)
        for i in range(len(tempX)):
            KXX = self.kernel(theta, tempX[i], tempX[i])
            KM = self.kernel(theta, tempX[i], self.MtrainX)
            sigma_t = np.diag(KXX - (KM @ KMM @ KM.T)).ravel()
            np.save(r'{}\{}.npy'.format(tmppath, i), sigma_t)
            del KXX, KM, sigma_t
        del tempX

        temp = []
        for i in range(split):
            temp.append(np.load(r'{}\{}.npy'.format(tmppath, i)))
        sigma_t = np.array(temp).ravel()
        print('Total Time = {}'.format(t2 - t1))
        rmtree(tmppath)
        var = sigma_t + sigma_n
        return var

    def train(self, train_data: array, ard=True, **params):
        """
        Training Steps
        1. Saves train_data in self._path folder as _train.dat
        2. Runs bisvm.exe with args -a -o -q -B '{}'.format(init_beta) -> process
        3. Get Parameters
        3.1 _Parameter text file consists of initial[0], final[1] and gradient[2].
            Get final params and store in self.params, aso results in self.results
        4. Extract mean and std of each dimension
        5. Import alphas
        6. Reconstruct Covariance for Off-bound SVs
        """
        self.ard = ard
        # 1
        self.trainX = train_data[:, :-1]
        self.trainy = train_data[:, -1]
        train_path = r'{}\_train.dat'.format(self.path_save)
        np.savetxt(train_path, train_data)
        self.ndim = train_data.shape[1] - 1
        # 2
        if bool(params):
            # cmd_ = params['cmd']["-a", "-c"]
            # C = params['params']['C']
            # eps = params['params']['Epsilon']
            beta = params['params']['Beta']
            # kaio = params['Kaio']
            # kappa_b = params['Kappa_b']
            # ard_ = params['params']['ARD']
            if ard:
                cmd = [r"{}".format(self.path_exe), "-a", "-c",
                       "-B", "{}".format(beta),
                       "-C", "{}".format(1000),
                       "-Cs", "100",
                       #    "-E", "{}".format(eps),
                       #    "-K", "{}".format(ard_),
                       # "-O", "{}".format(kaio),
                       "-A", "100",
                       "-As", "1000",
                       "-L", "10",
                       # "-S", "12",
                       r"{}".format(train_path)]
            else:
                cmd = [r"{}".format(self.path_exe), "-a", "-c",
                       "-B", "{}".format(beta),
                       # "-C", "{}".format(C),
                       # "-E", "{}".format(eps),
                       # "-K", "{}".format(ard),
                       # "-O", "{}".format(kaio),
                       # "-A", "{}".format(kappa_b),
                       # "-L", "5",
                       # "-S", "12",
                       r"{}".format(train_path)]
        else:
            cmd = [r"{}".format(self.path_exe), "-a", "-h", "-c",
                   "-B", "{}".format(self.init_beta),
                   r"{}".format(train_path)]
        # _check_call(cmd, stdout=_DEVNULL)
        _check_call(cmd)

        # 3
        f = open('{}\_train.dat_expert_1.log'.format(self.path_save))
        text = [i.strip() for i in f]
        f.close()

        extract_txt = lambda X: [s for s in text if "{}:".format(X) in s]
        extract_val = lambda X, idx: float(extract_txt(X)[idx].split(':')[1])

        # 4
        param_list: list = ["C", "Epsilon", "Beta", "Kaio", "Kappa_b"]
        idx1 = text.index("mean of each input dimension:") + 2
        idx2 = text.index("standard deviation of each input dimension:") + 2
        self.trainX_ = np.zeros(self.trainX.shape)
        for i in range(self.ndim):
            if ard:
                param_list.append("ARD parameter of dimension {}".format(i + 1))
            else:
                param_list.append("Kernel parameter KAI")
            self.mean.append(float(text[idx1].split('\t')[i]))
            self.std.append(float(text[idx2].split('\t')[i]))
            self.trainX_[:, i] = (self.trainX[:, i] - self.mean[i]) / self.std[i]
        self.mean.append(train_data[:, -1].mean())
        self.std.append(train_data[:, -1].std())
        res_list: list = ["EVIDENCE", "Variance of noise in original data", "The variance of SILF noise", ]
        # svm_list: list = ["on bound", "off bound", "none"]

        params = [extract_val(i, 1) for i in param_list]
        res = [extract_val(i, 0) for i in res_list]
        # svm = [extract_val(i, 0) for i in svm_list]
        self.params = dict(zip(param_list, params))
        self.results = dict(zip(res_list, res))
        # self.svm_results = dict(zip(svm_list, svm))
        self.evidence = self.results["EVIDENCE"]

        # 5 Used to calculate loss function for likelihood
        self.alpha = np.loadtxt('{}\_alpha.dat'.format(self.path_save))
        idx_off = np.argwhere(np.logical_and(0 < abs(self.alpha), abs(self.alpha) < self.params["C"])).ravel()
        idx_none = np.argwhere(abs(self.alpha) == 0)
        on_idx = np.arange(0, len(self.alpha))
        idx_on = np.array([i for i in on_idx if i not in idx_off and i not in idx_none])
        self.sv_list = {'off': idx_off, 'on': idx_on, 'none': idx_none}

        # 6
        if self.ndim == 1:
            KL = [self.params["ARD parameter of dimension 1"]]
        elif self.ndim == 2:
            if ard:
                KL = [self.params["ARD parameter of dimension 1"], self.params["ARD parameter of dimension 2"]]
            else:
                KL = [self.params["Kernel parameter KAI"]]
        theta = [self.params["Kaio"], KL, self.params["Kappa_b"]]
        self.MPsi = self.kernel(theta, self.trainX[idx_off, :], self.trainX[idx_off, :])

        self.MtrainX = self.trainX_[idx_off, :]  # Normalized
        # self.MtrainX = self.trainX[idx_off, :]

    def predict(self, testX, return_var=False):

        # Normalize (no -i and -o) testX
        temp = np.zeros(testX.shape)
        for i in range(testX.shape[1]):
            temp[:, i] = (testX[:, i] - self.mean[i]) / self.std[i]
        testX = temp

        if self.ndim == 1:
            KL = [self.params["ARD parameter of dimension 1"]]
        elif self.ndim == 2:
            if self.ard:
                KL = [self.params["ARD parameter of dimension 1"], self.params["ARD parameter of dimension 2"]]
            else:
                KL = [self.params["Kernel parameter KAI"]]
        theta = [self.params["Kaio"], KL, self.params["Kappa_b"]]
        K = self.kernel(theta, self.trainX_, testX)  # Normalized
        # K = self.kernel(theta, self.trainX, testX)
        mean = (np.dot(self.alpha.reshape(1, -1), K)).ravel()
        # # Normalized
        if return_var:
            var = self._variance(theta, testX)
            return (mean * self.std[-1] + self.mean[-1]), var * self.std[-1] ** 2
        else:
            return mean * self.std[-1] + self.mean[-1]




