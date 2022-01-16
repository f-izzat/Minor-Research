from utilities import *

from dataclasses import dataclass as _dataclass
from scipy.optimize import minimize as _minimize
from optuna import create_study as _create_study, samplers as _samplers, logging as _logging
from optuna.pruners import MedianPruner as _MedianPruner
from cvxopt import matrix as _matrix, solvers as _solvers
import mosek as _mosek


@_dataclass(frozen=True)
class Parameter:
    C: float
    theta: list
    eps: float
    ktype: str
    lossfxn: int = 2


@_dataclass
class SVRInfo:
    trainX: array
    trainy: array
    search_space: dict
    svrtype: str = 'EpsilonSVR'
    optmethod: int = 1
    lossfxn: int = 2
    approxcv: bool = False
    n_fold: int = 5
    n_trial: int = 300

    # To be filled during training
    CVE: float = 0.0
    Model: Any = None
    Param = None


class EpsilonSVR:
    def __init__(self, param: Parameter) -> None:
        self.param = param
        kernels = {'GAUSSIAN': Kernels.k_gaussian,
                   'MATERN32': Kernels.k_matern32,
                   'MATERN52': Kernels.k_matern52,
                   }
        self.kernel = kernels['{}'.format(param.ktype)]
        self.scaler = MinMaxScaler(clip=True)
        if param.lossfxn == 1:
            self.train = self._trainL1
        else:
            self.train = self._trainL2

        # - Filled when training
        self.Psi: array = None
        self.trainX_: array = None  # Scaled
        self.dual: array = None
        self.slackp: array = None
        self.slackm: array = None
        self.sv_idx: array = None
        self.X_sv: array = None
        self.rho: float = 0.0

    @staticmethod
    def normalize(X: array, mean: float, std: float) -> array:
        return (X - mean) / std

    def _trainL1(self, X: array, y: array, scale: bool = True) -> None:
        # Normalize data f:
        if scale:
            self.scaler.fit(X)
            X_ = self.scaler.transform(X)
            y_ = y
        else:
            X_ = X
            y_ = y

        Psi = self.kernel(self.param.theta, X_, X_)
        n_train = len(X_)

        G = _matrix(np.vstack((-np.eye(2 * n_train), np.eye(2 * n_train))), tc='d')
        h = _matrix(np.vstack((np.zeros(shape=[2 * n_train, 1]), np.ones(shape=[2 * n_train, 1]) * self.param.C)))
        A = _matrix(np.hstack((np.ones(shape=[1, n_train]), -np.ones(shape=[1, n_train]))), tc='d')
        b = _matrix(0.0)

        P = _matrix(np.hstack((np.vstack((Psi, -Psi)), np.vstack((-Psi, Psi)))))
        q = _matrix(np.vstack((self.param.eps - y_.reshape(-1, 1), self.param.eps + y_.reshape(-1, 1))))

        _solvers.options['mosek'] = {_mosek.iparam.log: 0}
        _solvers.options['show_progress'] = False
        sol = _solvers.qp(P, q, G, h, A, b)
        alpha_ = np.ravel(np.array(sol['x']))
        alpha_p = alpha_[:n_train]
        alpha_m = alpha_[n_train:2 * n_train]
        self.dual = alpha_p - alpha_m
        self.sv_idx = np.argwhere(abs(self.dual) > float(1e-12)).ravel()
        self.X_sv = X_[self.sv_idx, :]
        sv_mid_i = \
        np.where(min((abs(abs(self.dual) - (self.param.C / 2)))) == (abs(abs(self.dual) - (self.param.C / 2))))[0][0]
        self.rho = y_[sv_mid_i] - (self.param.eps * np.sign(self.dual[sv_mid_i])) - np.dot(self.dual[self.sv_idx],
                                                                                           Psi[self.sv_idx, sv_mid_i])
        self.slackp = (y - (self.predict(X_, scale=False) + self.param.eps)).ravel()
        self.slackm = ((self.predict(X_, scale=False) - self.param.eps) - y).ravel()
        self.Psi = Psi

    def _trainL2(self, X: array, y: array, scale: bool = True) -> None:
        # Normalize data f:
        if scale:
            self.scaler.fit(X)
            X_ = self.scaler.transform(X)
            y_ = y
        else:
            X_ = X
            y_ = y

        Psi = self.kernel(self.param.theta, X_, X_)
        np.fill_diagonal(Psi, Psi.diagonal() + (1 / self.param.C))
        n_train = len(X_)

        G = _matrix(np.vstack((-np.eye(2 * n_train), np.eye(2 * n_train))), tc='d')
        h = _matrix(np.vstack((np.zeros(shape=[2 * n_train, 1]), np.ones(shape=[2 * n_train, 1]) * self.param.C)))
        A = _matrix(np.hstack((np.ones(shape=[1, n_train]), -np.ones(shape=[1, n_train]))), tc='d')
        b = _matrix(0.0)

        P = _matrix(np.hstack((np.vstack((Psi, -Psi)), np.vstack((-Psi, Psi)))))
        q = _matrix(np.vstack((self.param.eps - y_.reshape(-1, 1), self.param.eps + y_.reshape(-1, 1))))

        _solvers.options['mosek'] = {_mosek.iparam.log: 0}
        _solvers.options['show_progress'] = False
        sol = _solvers.qp(P, q, G, h, A, b)
        alpha_ = np.ravel(np.array(sol['x']))
        alpha_p = alpha_[:n_train]
        alpha_m = alpha_[n_train:2 * n_train]

        self.dual = alpha_p - alpha_m
        self.sv_idx = np.argwhere(abs(self.dual) > float(1e-12)).ravel()
        self.X_sv = X_[self.sv_idx, :]
        sv_mid_i = \
        np.where(min((abs(abs(self.dual) - (self.param.C / 2)))) == (abs(abs(self.dual) - (self.param.C / 2))))[0][0]
        self.rho = y_[sv_mid_i] - (self.param.eps * np.sign(self.dual[sv_mid_i])) - np.dot(self.dual[self.sv_idx],
                                                                                           Psi[self.sv_idx, sv_mid_i])
        self.Psi = Psi

    def predict(self, testX: array, path_save=None, scale: bool = True) -> array:
        if scale:
            testX_ = self.scaler.transform(testX)
        else:
            testX_ = testX

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
                K: array = self.kernel(self.param.theta, self.X_sv, tempX[i])
                pred = (np.dot(self.dual[self.sv_idx].reshape(1, -1), K)).ravel() + float(self.rho)
                np.save(r'{}\{}.npy'.format(tmppath, i), pred)
            del tempX
            temp = []
            for i in range(split):
                temp.append(np.load(r'{}\{}.npy'.format(tmppath, i)))
            pred_ = np.array(temp).ravel()
            rmtree(tmppath)
            return pred_
        else:
            K: array = self.kernel(self.param.theta, self.X_sv, testX_)
            return (np.dot(self.dual[self.sv_idx].reshape(1, -1), K)).ravel() + float(self.rho)


class LSSVR:
    kernels = {'RBF': Kernels.k_rbf,
               'GAUSSIAN': Kernels.k_gaussian,
               'GAUSSIAN4': Kernels.k_gaussian4,
               'MATERN32': Kernels.k_matern32,
               'MATERN52': Kernels.k_matern52,
               }

    def __init__(self, param: Parameter):
        self.param = param
        self.trainX_ = None
        self.trainY_ = None
        self.dual: array = None
        self.rho: float = 0.0

        self._kernel = LSSVR.kernels['{}'.format(param.ktype)]
        self.Psi_ = None
        self._scaler = MinMaxScaler(clip=True)

    def _smoother(self, K):
        # Smoothing + Double Smoothing
        Z = np.linalg.pinv(self.Psi_)
        c = (Z.T.sum(axis=1)).sum()
        J = np.ones(Z.shape)/c
        J1 = np.ones([len(K), len(Z)])/c
        smoother = (K @ (Z-(Z@J@Z))) + (J1@Z)

        bparam =deepcopy(self.param)
        bparam.ktype = 'GAUSSIAN4'
        modelb = LSSVR(param=bparam)
        modelb.train(trainX=self.trainX_.copy(), trainy=self.trainY_.copy())
        biascorr = (smoother - np.eye(len(self.trainX_))) @ modelb.predict(self.trainX_.copy())
        return smoother, biascorr

    def train(self, trainX: array, trainy: array, scale=True):
        self._scaler.fit(X=trainX, y=trainy)
        if scale:
            trainX_ = self._scaler.transform(trainX)
        else:
            trainX_ = trainX

        Psi = self._kernel(self.param.theta, trainX_, trainX_)
        self.Psi_ = Psi + np.diag(np.ones(len(Psi)) * (1 / self.param.C))

        A = np.vstack([np.append(0, np.ones([1, len(trainX_)])), np.hstack([np.ones([len(trainX_), 1]), self.Psi_])])
        b = np.append(0, trainy).reshape(-1, 1)
        x_sol = np.linalg.solve(A, b)
        self.rho = x_sol[0].item()
        self.dual = x_sol[1:].ravel()

        self.trainX_ = trainX.copy()
        self.trainy_ = trainy.copy()

    def predict(self, testX: array, return_std=False, scale=True, path_save=None):
        trainX_ = self._scaler.transform(self.trainX_)
        if scale:
            testX_ = self._scaler.transform(testX)
        else:
            testX_ = testX
        
        if return_std:
            # Returns Pointwise Confidence Interval
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
                    K: array = self._kernel(self.param.theta, trainX_, tempX[i])
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
                K: array = self._kernel(self.param.theta, trainX_, testX_)
                smooth = self._smoother(K)

                return (np.dot(self.dual.reshape(1, -1), K)).ravel() + float(self.rho)
        else:
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
                    K: array = self._kernel(self.param.theta, trainX_, tempX[i])
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
                K: array = self._kernel(self.param.theta, trainX_, testX_)
                S, BC = self._smoother(K)
                
                return (np.dot(self.dual.reshape(1, -1), K)).ravel() + float(self.rho)            



class Tune:
    _space: dict = {'lb': {'C': 1, 'theta': 0.001, 'eps': 0.0001},
                    'ub': {'C': 100, 'theta': 1, 'eps': 0.01},
                    'x0': {'C': 10, 'theta': 1, 'eps': 0.001},
                    'kernel_type': 'MATERN32',
                    'lossfxn': 2,
                    'ARD': True}
    _best: Optional[EpsilonSVR]

    def __init__(self, svr_info: SVRInfo) -> None:
        self.t_info = svr_info
        self.train_data = np.hstack((svr_info.trainX, svr_info.trainy.reshape(-1, 1)))

    def callback(self, study, trial):
        if study.best_trial.number == trial.number:
            self.t_info.Model = Tune._best
            self.t_info.Model.train(self.train_data[:, :-1].copy(), self.train_data[:, -1].copy())

    def _approxcv(self, model: EpsilonSVR):
        model.train(self.train_data[:, :-1].copy(), self.train_data[:, -1].copy())
        sv = model.sv_idx
        alphas = model.dual[sv].ravel()
        # Ksv = model.Psi[np.ix_(sv, sv)]
        # tmp1 = np.ones([Ksv.shape[0], 1])
        # tmp2 = np.append(np.ones([1, Ksv.shape[1]]), 0).reshape(1, -1)
        # Kbsv = np.vstack([np.hstack([Ksv, tmp1]), tmp2])
        #
        # # # Smoothed Span Estimate
        # eta = 0.1
        # D = np.diag(eta / alphas)
        # tmp1 = np.zeros([D.shape[0], 1])
        # tmp2 = np.append(np.zeros([1, D.shape[1]]), 0).reshape(1, -1)
        # Db = np.vstack([np.hstack([D, tmp1]), tmp2])
        # Stil = (1 / (np.linalg.pinv(Kbsv + Db)).diagonal()) - Db.diagonal()

        if self.t_info.lossfxn == 1:
            ubd = np.argwhere(np.logical_and(0 < abs(alphas), abs(alphas) < model.param.C)).ravel()
            bd = np.argwhere(abs(alphas) == model.param.C).ravel()
            Stil = np.zeros(len(ubd) + len(bd), )
            if len(ubd) != 0:
                Ksv = model.Psi[np.ix_(sv[ubd], sv[ubd])]
                tmp1 = np.ones([Ksv.shape[0], 1])
                tmp2 = np.append(np.ones([1, Ksv.shape[1]]), 0).reshape(1, -1)
                Kbsv = np.vstack([np.hstack([Ksv, tmp1]), tmp2])
                uStil = (1 / np.linalg.inv(Kbsv).diagonal()).ravel()
                Stil[:len(ubd)] = uStil
            if len(bd) != 0:
                Ksv: array = model.Psi[np.ix_(sv[bd], sv[bd])]
                for i in range(len(bd)):
                    Stil[len(ubd):i] = (Ksv.diagonal()[i]).item() - \
                                       (Ksv[:, i].reshape(1, -1) @ np.linalg.inv(Ksv) @ Ksv[:, i].reshape(-1, 1)).item()
            return abs(
                model.param.eps + (np.sum(alphas * Stil[:-1]) + np.sum(model.slackp[sv] + model.slackm[sv])) * 1 / len(
                    sv))
        else:
            Ksv = model.Psi[np.ix_(sv, sv)]
            tmp1 = np.ones([Ksv.shape[0], 1])
            tmp2 = np.append(np.ones([1, Ksv.shape[1]]), 0).reshape(1, -1)
            Kbsv = np.vstack([np.hstack([Ksv, tmp1]), tmp2])
            Stil = (1 / np.linalg.inv(Kbsv).diagonal()).ravel()
            return abs(model.param.eps + np.sum(alphas * Stil[:-1]) * 1 / len(sv))

    def _cross_validate(self, model: Optional[EpsilonSVR]):
        rmse = lambda y1, y2: np.sqrt(np.average((y1 - y2) ** 2))
        kf = KFold(n_splits=self.t_info.n_fold)
        error = []
        X, y = self.t_info.trainX.copy(), self.t_info.trainy.copy()
        for tridx, teidx in kf.split(X, y):
            model.train(X[tridx], y[tridx])
            pred = model.predict(X[teidx])
            error.append(rmse(y[teidx], pred))
        cv_error = (1 / self.t_info.n_fold) * sum(error)
        return cv_error

    def objective1(self, trial):
        if bool(self.t_info.search_space):
            if self.train_data.shape[1] - 1 == 1 or self.t_info.search_space['kernel_type'] == 'RBF':
                theta = trial.suggest_uniform('theta', self.t_info.search_space['lb']['theta'],
                                              self.t_info.search_space['ub']['theta'])
                theta = [theta]
            elif self.train_data.shape[1] - 1 == 2 and self.t_info.search_space['kernel_type'] != 'RBF':
                theta1 = trial.suggest_uniform('theta1', self.t_info.search_space['lb']['theta'],
                                               self.t_info.search_space['ub']['theta'])
                if self.t_info.search_space['ARD']:
                    theta2 = trial.suggest_uniform('theta2', self.t_info.search_space['lb']['theta'],
                                                   self.t_info.search_space['ub']['theta'])
                    theta = [theta1, theta2]
                else:
                    theta = [theta1, theta1]

            C = trial.suggest_loguniform('C', self.t_info.search_space['lb']['C'], self.t_info.search_space['ub']['C'])

            if self.t_info.svrtype == 'EpsilonSVR':
                eps = trial.suggest_loguniform('eps', self.t_info.search_space['lb']['eps'],
                                               self.t_info.search_space['ub']['eps'])
                model = EpsilonSVR(Parameter(C=C, theta=theta, eps=eps,
                                             ktype=self.t_info.search_space['kernel_type'],
                                             lossfxn=self.t_info.lossfxn))
            else:
                model = LSSVR(Parameter(C=C, theta=theta, eps=0.0,
                                        ktype=self.t_info.search_space['kernel_type']))

        else:
            if self.train_data.shape[1] - 1 == 1 or Tune._space['kernel_type'] == 'RBF':
                theta = trial.suggest_uniform('theta', Tune._space['lb']['theta'],
                                              Tune._space['ub']['theta'])
                theta = [theta]
            elif self.train_data.shape[1] - 1 == 2 and Tune._space['kernel_type'] != 'RBF':
                theta1 = trial.suggest_uniform('theta1', Tune._space['lb']['theta'],
                                               Tune._space['ub']['theta'])
                if Tune._space['ARD']:
                    theta2 = trial.suggest_uniform('theta2', Tune._space['lb']['theta'],
                                                   Tune._space['ub']['theta'])
                    theta = [theta1, theta2]
                else:
                    theta = [theta1, theta1]

            C = trial.suggest_loguniform('C', Tune._space['lb']['C'], Tune._space['ub']['C'])
            if self.t_info.svrtype == 'EpsilonSVR':
                eps = trial.suggest_loguniform('eps', Tune._space['lb']['eps'],
                                               Tune._space['ub']['eps'])

                model = EpsilonSVR(Parameter(C=C, theta=theta, eps=eps,
                                             ktype=Tune._space['kernel_type'],
                                             lossfxn=Tune._space['lossfxn']))
            else:
                model = LSSVR(Parameter(C=C, theta=theta, eps=0.0,
                                        ktype=Tune._space['kernel_type']))

        if self.t_info.approxcv:
            cv_error = self._approxcv(model)
        else:
            cv_error = self._cross_validate(model)
        Tune._best = model
        return cv_error

    def objective2(self, x0, ktype: str):
        """
        LBFGSB Method (C, theta,eps)
        """
        model = EpsilonSVR(Parameter(C=x0[0], theta=[x0[1]], eps=x0[2], ktype=ktype))
        cv_error = self._cross_validate(model)
        return cv_error

    def run(self, params: Optional[dict] = None, log=True):
        if self.t_info.optmethod == 1:
            """ Optuna Method """
            if not log:
                _logging.set_verbosity(_logging.WARNING)
            study = _create_study(direction='minimize',
                                  sampler=_samplers.TPESampler(n_startup_trials=25, n_ei_candidates=100),
                                  pruner=_MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10))
            if params is not None:
                study.enqueue_trial(params)
            study.optimize(self.objective1, n_trials=self.t_info.n_trial, callbacks=[self.callback])
            self.t_info.CVE = study.best_value
            self.t_info.Param = study.best_params
        else:
            """ LBFGSB Method """
            if bool(self.t_info.search_space):
                lb = np.array(tuple(self.t_info.search_space['lb'].values())).reshape(-1, 1)
                ub = np.array(tuple(self.t_info.search_space['ub'].values())).reshape(-1, 1)
                x0 = np.array(tuple(self.t_info.search_space['x0'].values())).reshape(-1, 1)
                bound = np.hstack((lb, ub))
                res = _minimize(self.objective2, x0, method='L-BFGS-B',
                                options={'maxiter': 1000, 'eps': 1e-12, 'disp': True},
                                args=(self.t_info.search_space['kernel_type']),
                                bounds=bound)
                self.t_info.Param = res.x
                self.t_info.CVE = res.fun
                self.t_info.Model = Tune._best
                self.t_info.Model.train(self.train_data[:, :-1].copy(), self.train_data[:, -1].copy())
            else:
                lb = np.array(tuple(Tune._space['lb'].values())).reshape(-1, 1)
                ub = np.array(tuple(Tune._space['ub'].values())).reshape(-1, 1)
                x0 = np.array(tuple(Tune._space['x0'].values())).reshape(-1, 1)
                bound = np.hstack((lb, ub))
                res = _minimize(self.objective2, x0, method='L-BFGS-B',
                                options={'maxiter': 1000, 'eps': 1e-12, 'disp': True},
                                args=(Tune._space['kernel_type']),
                                bounds=bound)
                self.t_info.Param = res.x
                self.t_info.CVE = res.fun
                self.t_info.Model.train(self.train_data[:, :-1].copy(), self.train_data[:, -1].copy())


