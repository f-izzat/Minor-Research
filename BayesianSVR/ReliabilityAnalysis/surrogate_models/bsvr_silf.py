from utilities import *
from optuna import create_study as _create_study, samplers as _samplers, logging as _logging
from optuna.pruners import MedianPruner as _MedianPruner
from subprocess import check_call as _check_call, DEVNULL as _DEVNULL
from scipy.special import erf as _erf

"""
cmd list:
  -a switch on automatic relevance determination.
  -v activate the verbose mode to display messages. 
  -r initialize the hyperparameters randomly.
  -u use as unified sequential minimal optimization.
  -s indicate standard SVM training by usmo.
  -o keep the outputs original.
  -i keep the inputs original.
  -q switch off quick training mode.
  -b choose beta heuristically.
  -w use BFGS with Wolfe-Powell line search as optimization package.
  -h use the second heuristic.
  -m [n]	mixture of n experts to be trained.
  -C [c]	initialize trade-off C as the value c.
  -Cs [cs]	initializes superior C at the value cs.
  -Ci [ci]	initializes inferior C at the value ci.
  -B [beta]	set Beta at the value beta.
  -E [eps]	initializes Epsilon at the value eps.
  -Es [es]	initializes superior Epsilon at the value es.
  -Ei [ei]	initializes inferior Epsilon at the value ei.
  -K [kai]	initializes Kai at the value kai.
  -Ks [ks]	initializes superior Kai at the value ks.
  -Ki [ki]	initializes inferior Kai at the value ki.
  -O [kaio]	initializes KaiO at the value kaio.
  -Os [os]	initializes superior kaiO at the value os.
  -Oi [oi]	initializes inferior kaiO at the value oi.
  -A [ab]	initializes kappA_b at the value ab.
  -As [as]	initializes superior kappA_b at the value as.
  -Ai [ai]	initializes inferior kappA_b at the value ai.
  -P [p]	choose Polynomial kernel with the order p.
  -S [sec]	set time limit in seconds at the value sec.
  -L [loop]	set evaluation limit at the value loop.
  -T [tol]	set Tolerance at the value tol.
  -R [r]	set Regularization factor at r (default r=0.1).

"""


@dataclass
class BSVRInfo:
    root_path: str
    exe_path: str = r'D:\PYTHON\.PROJECTS\BayesianSVR\EXECUTE2\bisvm.exe'
    n_trial: int = 100
    space: dict = field(default_factory=dict)


class BSVR_SILF:
    _info: BSVRInfo

    def __init__(self, info: BSVRInfo):
        BSVR_SILF._info = info

        if not exists(info.root_path):
            makedirs(info.root_path)

        self.results: dict = {}
        self._kernel = Kernels.k_silf
        self._trainX: array = None
        self._alpha: array = None
        self._svlist: dict = {}
        self._params: dict = {}
        self._mean: list = []
        self._std: list = []
        self._ndim: int = 0

    def _post_process(self, trainX, trainy):
        f = open('{}\_train.dat_expert_1.log'.format(BSVR_SILF._info.root_path))
        text = [i.strip() for i in f]
        f.close()
        extract_txt = lambda X: [s for s in text if "{}:".format(X) in s]
        extract_val = lambda X, idx: float(extract_txt(X)[idx].split(':')[1])
        param_list: list = ["C", "Epsilon", "Beta", "Kaio", "Kappa_b"]

        idx1 = text.index("mean of each input dimension:") + 2
        idx2 = text.index("standard deviation of each input dimension:") + 2

        self._trainX = np.zeros(trainX.shape)
        for i in range(self._ndim):
            param_list.append("ARD parameter of dimension {}".format(i + 1))
            self._mean.append(float(text[idx1].split('\t')[i]))
            self._std.append(float(text[idx2].split('\t')[i]))
            self._trainX[:, i] = (trainX[:, i] - self._mean[i]) / self._std[i]

        self._mean.append(trainy.mean())
        self._std.append(trainy.std())
        res_list: list = ["EVIDENCE", "Variance of noise in original data", "The variance of SILF noise", ]

        params = [extract_val(i, 1) for i in param_list]
        res = [extract_val(i, 0) for i in res_list]
        self._params = dict(zip(param_list, params))
        self.results = dict(zip(res_list, res))

        # Used to calculate loss function for likelihood
        self._alpha = np.loadtxt('{}\_alpha.dat'.format(BSVR_SILF._info.root_path))
        idx_off = np.argwhere(np.logical_and(0 < abs(self._alpha), abs(self._alpha) < self._params["C"])).ravel()
        idx_none = np.argwhere(abs(self._alpha) == 0)
        on_idx = np.arange(0, len(self._alpha))
        idx_on = np.array([i for i in on_idx if i not in idx_off and i not in idx_none])
        self._svlist = {'off': idx_off, 'on': idx_on, 'none': idx_none}

    def _variance(self, theta, testX):
        print("Computing Variance...")
        C, eps, beta = self._params["C"], self._params["Epsilon"], self._params["Beta"]
        Zs = (2 * eps * (1 - beta)) + (2 * np.sqrt((np.pi * beta * eps) / C).item() * _erf(np.sqrt(C * beta * eps))) + (
                (2 / C) * np.exp(-C * beta * eps).item())
        t1 = ((eps ** 3) * (1 - beta) ** 3) / 3
        t2 = (np.sqrt((np.pi * beta * eps) / C).item()) * ((2 * beta * eps / C) + (eps ** 2) * (1 - beta) ** 2) * _erf(
            np.sqrt(C * beta * eps)).item()
        t3 = (4 * beta * eps ** 2) * (1 - beta) / C
        t4 = ((((eps ** 2) * (1 - beta) ** 2) / C) + ((2 * eps * (1 + beta)) / C ** 2) + (2 / C ** 3)) * np.exp(
            -C * beta * eps).item()
        sigma_n = (2 / Zs) * (t1 + t2 + t3 + t4)

        X_off = self._trainX[self._svlist['off'], :]
        KMM = self._kernel(theta, X_off, X_off)
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

        tmppath = r'{}\temp'.format(BSVR_SILF._info.root_path)
        if exists(tmppath):
            rmtree(tmppath)

        makedirs(tmppath)
        tempX = np.array_split(testX, split)
        for i in range(len(tempX)):
            KXX = self._kernel(theta, tempX[i], tempX[i])
            KM = self._kernel(theta, tempX[i], X_off)
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

    def train(self, trainX: array, trainy: array, cmdlist, log=True):

        train_path = r'{}\_train.dat'.format(BSVR_SILF._info.root_path)
        trainX = check_2d(trainX)
        np.savetxt(train_path, np.hstack((trainX, trainy.reshape(-1, 1))))
        self._ndim = trainX.shape[1]
        cmd = [BSVR_SILF._info.exe_path, '-a', '-c']
        cmd.extend(cmdlist)
        cmd.append(train_path)
        if log:
            _check_call(cmd)
        else:
            _check_call(cmd, stdout=_DEVNULL)

        self._post_process(trainX, trainy)

    def predict(self, testX, return_var=False):
        # Normalize (no -i and -o) testX
        temp = np.zeros(testX.shape)
        for i in range(testX.shape[1]):
            temp[:, i] = (testX[:, i] - self._mean[i]) / self._std[i]
        testX = temp
        KL = [self._params["ARD parameter of dimension {}".format(i+1)] for i in range(self._ndim)]
        theta = [self._params["Kaio"], KL, self._params["Kappa_b"]]
        K = self._kernel(theta, self._trainX, testX)  # Normalized
        mean = (np.dot(self._alpha.reshape(1, -1), K)).ravel()
        # # Normalized
        if return_var:
            var = self._variance(theta, testX)
            return (mean * self._std[-1] + self._mean[-1]), var * self._std[-1] ** 2
        else:
            return mean * self._std[-1] + self._mean[-1]


class Tune:
    _space: dict = {'lb': {'Beta': 0.5, 'C': 1, 'eps': 0.0001, 'theta': 0.1, 'Kaio': 0.001, 'Kappa_b': 1},
                    'ub': {'Beta': 0.9, 'C': 100, 'eps': 0.01, 'theta': 1, 'Kaio': 1, 'Kappa_b': 100}}
    _best: BSVR_SILF
    info: BSVRInfo

    def __init__(self, info: BSVRInfo) -> None:
        Tune.info = info

        self.train_data: array = None
        self.model = None
        self.params = None

    def callback(self, study, trial):
        if study.best_trial.number == trial.number:
            self.model = Tune._best

    @staticmethod
    def objective(trial, trainX: array, trainy: array):
        beta = trial.suggest_uniform('beta', Tune._space['lb']['Beta'], Tune._space['ub']['Beta'])
        # C = trial.suggest_uniform('C', Tune._space['lb']['C'], Tune._space['ub']['C'])
        # eps = trial.suggest_uniform('eps', Tune._space['lb']['Epsilon'], Tune._space['ub']['Epsilon'])
        # kai = trial.suggest_uniform('ard', Tune._space['lb']['theta'], Tune._space['ub']['theta'])
        # kaio = trial.suggest_uniform('kaio', Tune._space['lb']['Kaio'], Tune._space['ub']['Kaio'])
        # kappa_b = trial.suggest_uniform('kappa_b', Tune._space['lb']['Kappa_b'], Tune._space['ub']['Kappa_b'])
        # cmd = ['-C', '{}'.format(C), '-E', '{}'.format(eps), '-B', '{}'.format(beta), 'K', '{}'.format(kai),
        #        '-A', '{}'.format(kappa_b), '-O', '{}'.format(kaio)]
        cmd = ['-B', '{}'.format(beta)]

        model = BSVR_SILF(Tune.info)
        model.train(trainX, trainy, cmdlist=cmd)
        Tune._best = model
        return model.results["EVIDENCE"]

    def run(self, trainX, trainy, params=None, log=True):
        if not log:
            _logging.set_verbosity(_logging.WARNING)
        study = _create_study(direction='minimize',
                              sampler=_samplers.TPESampler(n_startup_trials=25),
                              pruner=_MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10))

        if params is not None:
            study.enqueue_trial(params)
        study.optimize(lambda trial: self.objective(trial, trainX, trainy), n_trials=Tune.info.n_trial,
                       callbacks=[self.callback])
        self.params = study.best_params
