from utilities import *
from scipy import stats as _stats
from surrogate_models.bsvr_silf import Tune as _Tune, BSVRInfo as _BSVRInfo


@dataclass
class AdaptInfo:
    root_path: str
    rmodel: Any
    initialX: array
    data_range: list
    learnfxn: str = 'ufxn'
    n_mcs: int = int(1e4)
    max_iter: int = 100


class ABSVR:
    _info: AdaptInfo
    _trainX: array
    _trainy: array
    _param: dict
    _plot: dict

    def __init__(self, info: AdaptInfo) -> None:
        ABSVR._info = info
        ABSVR._plot = {'xlabel': 'Mach', 'ylabel': 'Flutter Speed Index', 'zlabel': 'Damping Coefficient',
                       'title': 'ABSVR-MCS'}

        print('Initialize...')
        self.iter = 0

        if ABSVR._info.learnfxn == 'ufxn':
            self.lfxn = self._u_function
        elif ABSVR._info.learnfxn == 'erf':
            self.lfxn = self._exp_rf
        else:
            self.lfxn = self._exp_ff

        self._generate_folder()
        self.rmodel = info.rmodel
        ABSVR._trainX = info.initialX.copy()
        ABSVR._trainy = self.rmodel.predict(ABSVR._trainX.copy(), model='svr')
        init_model = self._generate_model(n_trial=200, reload=False)
        self._save_trajectory(init_model, init=True)
        self.iter += 1
        self._main_loop(init_model)

        # Updated Variables

        self._main_loop(init_model)

    @staticmethod
    def _probability(X):
        n_fail = np.sum(X.ravel() <= 0)
        return n_fail / len(X)

    @staticmethod
    def _exp_ff(mean: array, var: array, a=0, plot=False):
        """ Expected Feasibility Function """
        _eps = 2 * var
        sig = np.sqrt(var)
        term1 = (a - mean) / sig
        term2 = (a - _eps - mean) / sig
        term3 = (a + _eps - mean) / sig
        eff = (mean - a) * (2 * _stats.norm.cdf(term1) - _stats.norm.cdf(term2) - _stats.norm.cdf(term3)) \
              - sig * (2 * _stats.norm.pdf(term1) - _stats.norm.pdf(term2) - _stats.norm.pdf(term3)) \
              + (_stats.norm.cdf(term3) - _stats.norm.cdf(term2))
        idx = np.argmax(eff)
        print('Max EFF = {}'.format(max(eff).item()))
        cond = 1
        if max(eff) < 0.005:
            cond = 0
        if plot:
            return eff.ravel()
        else:
            return idx, cond, max(eff).item()

    @staticmethod
    def _exp_rf(mean: array, var: array, epsilon: float = 0.001, plot=False):
        mean_ = mean.ravel()
        std = np.sqrt(var).ravel()
        sign_ = np.sign(mean_).ravel()
        term1 = (-sign_ * mean_) * _stats.norm.cdf(-sign_ * (mean_ / std))
        term2 = std * _stats.norm.pdf(mean_ / std)
        erf = term1 + term2
        # Condition 1
        idx, val1 = np.argmax(erf), np.max(erf)
        print('max(ERF) = {}'.format(val1))
        # Condition 2
        Ppos = ABSVR._probability(mean_ + 1.96*std)
        Pneg = ABSVR._probability(mean_ - 1.96*std)
        P = ABSVR._probability(mean_)
        val2 = (Ppos - Pneg) / P
        print('Convergence Pf = {}'.format(val2))
        cond = 1
        if val1 < epsilon and val2 < 0.05:
            cond = 0
        print('Convergence State = {}'.format(cond))
        if plot:
            return erf.ravel()
        else:
            return idx, cond, val1

    @staticmethod
    def _u_function(mean: array, var: array, plot=False):
        # Rows of dist_ correspond to mcs_samples
        u = np.abs(mean) / np.sqrt(var)
        idx, val = np.argmin(u), np.min(u)
        print('min(U) = {}'.format(val))
        cond = 1
        if val >= 2:
            cond = 0
        if plot:
            return u.ravel()
        else:
            return idx, cond, val

    @staticmethod
    def _post_process(iterations):
        cov = np.loadtxt(r'{}/cov.dat'.format(ABSVR._info.root_path))
        prob = np.loadtxt(r'{}/prob.dat'.format(ABSVR._info.root_path))

        iterations = np.array(iterations)
        iterations[-1, 1] = iterations[-1, 1] - 1

        # ----------------  Learning Function ----------------#
        lvalue = np.loadtxt(r'{}/{}.dat'.format(ABSVR._info.root_path, ABSVR._info.learnfxn))
        fig1 = plt.figure(1)
        plt.plot(lvalue[:, 0], lvalue[:, -1], c='r')
        plt.scatter(lvalue[iterations[:, -1], 0], lvalue[iterations[:, -1], -1], c='k', s=10, marker="+")
        plt.xlabel('Inner Iterations')
        plt.ylabel('{} Value'.format(ABSVR._info.learnfxn.upper()))
        plt.savefig(r'{}/{}.png'.format(ABSVR._info.root_path, ABSVR._info.learnfxn))
        plt.close(fig1)

        fig2 = plt.figure(2)
        plt.plot(cov[:, 0], cov[:, -1], c='r')
        plt.scatter(cov[iterations[:, -1], 0], cov[iterations[:, -1], -1], c='k', s=10, marker="+")
        plt.xlabel('Inner Iterations')
        plt.ylabel('Coefficient of Variation')
        plt.savefig(r'{}/cov.png'.format(ABSVR._info.root_path))
        plt.close(fig2)

        fig3 = plt.figure(3)
        plt.plot(prob[:, 0], prob[:, -1], c='r')
        plt.scatter(prob[iterations[:, -1], 0], prob[iterations[:, -1], -1], c='k', s=10, marker="+")
        plt.xlabel('Inner Iterations')
        plt.ylabel('Failure Probability')
        plt.savefig(r'{}/probability.png'.format(ABSVR._info.root_path))
        plt.close(fig3)

    @staticmethod
    def _generate_mcs():
        print('Generating Monte-Carlo Simulations')
        n_ftr = ABSVR._info.initialX.shape[1]
        mcs = np.random.rand(ABSVR._info.n_mcs, n_ftr)
        min_, max_ = ABSVR._info.data_range[0], ABSVR._info.data_range[1]
        for i in range(mcs.shape[1]):
            mcs[:, i] = (max_[i] - min_[i]) * ((mcs[:, i] - min(mcs[:, i]))/(max(mcs[:, i]) - min(mcs[:, i]))) + min_[i]
        return mcs

    def _generate_folder(self):
        # # Create Files for storage
        tmp = np.array([])
        if not exists(r'{}\added.dat'.format(ABSVR._info.root_path)):
            np.savetxt(r'{}\added.dat'.format(ABSVR._info.root_path), tmp)
        else:
            remove(r'{}\added.dat'.format(ABSVR._info.root_path))
            np.savetxt(r'{}\added.dat'.format(ABSVR._info.root_path), tmp)

        if not exists(r'{}\cov.dat'.format(ABSVR._info.root_path)):
            np.savetxt(r'{}\cov.dat'.format(ABSVR._info.root_path), tmp)
        else:
            remove(r'{}\cov.dat'.format(ABSVR._info.root_path))
            np.savetxt(r'{}\cov.dat'.format(ABSVR._info.root_path), tmp)

        if not exists(r'{}\prob.dat'.format(ABSVR._info.root_path)):
            np.savetxt(r'{}\prob.dat'.format(ABSVR._info.root_path), tmp)
        else:
            remove(r'{}\prob.dat'.format(ABSVR._info.root_path))
            np.savetxt(r'{}\prob.dat'.format(ABSVR._info.root_path), tmp)

        if not exists(r'{}\ContourPlot\Mean'.format(ABSVR._info.root_path)):
            makedirs(r'{}\ContourPlot\Mean'.format(ABSVR._info.root_path))
            makedirs(r'{}\ContourPlot\Variance'.format(ABSVR._info.root_path))
            makedirs(r'{}\ContourPlot\{}'.format(ABSVR._info.root_path, self._info.learnfxn.upper()))
        else:
            rmtree(r'{}\ContourPlot\Mean'.format(ABSVR._info.root_path))
            makedirs(r'{}\ContourPlot\Mean'.format(ABSVR._info.root_path))
            makedirs(r'{}\ContourPlot\Variance'.format(ABSVR._info.root_path))
            makedirs(r'{}\ContourPlot\{}'.format(ABSVR._info.root_path, self._info.learnfxn.upper()))

        if not exists(r'{}\Iterations'.format(ABSVR._info.root_path)):
            makedirs(r'{}\Iterations'.format(ABSVR._info.root_path))
        else:
            rmtree(r'{}\Iterations'.format(ABSVR._info.root_path))
            makedirs(r'{}\Iterations'.format(ABSVR._info.root_path))

        if not exists(r'{}\ReferencePlot'.format(ABSVR._info.root_path)):
            makedirs(r'{}\ReferencePlot'.format(ABSVR._info.root_path))
        else:
            rmtree(r'{}\ReferencePlot'.format(ABSVR._info.root_path))
            makedirs(r'{}\ReferencePlot'.format(ABSVR._info.root_path))

        if not exists(r'{}\SurfacePlot'.format(ABSVR._info.root_path)):
            makedirs(r'{}\SurfacePlot'.format(ABSVR._info.root_path))
        else:
            rmtree(r'{}\SurfacePlot'.format(ABSVR._info.root_path))
            makedirs(r'{}\SurfacePlot'.format(ABSVR._info.root_path))

        if not exists(r'{}\{}.dat'.format(ABSVR._info.root_path, ABSVR._info.learnfxn)):
            np.savetxt(r'{}\{}.dat'.format(ABSVR._info.root_path, ABSVR._info.learnfxn), tmp)
        else:
            remove(r'{}\{}.dat'.format(ABSVR._info.root_path, ABSVR._info.learnfxn))
            np.savetxt(r'{}\{}.dat'.format(ABSVR._info.root_path, ABSVR._info.learnfxn), tmp)

    def _generate_model(self, n_trial, reload=True):
        model_info = _BSVRInfo(root_path=r'{}\Iterations\{}'.format(ABSVR._info.root_path, self.iter), n_trial=n_trial)
        tune = _Tune(model_info)
        if reload:
            tune.run(ABSVR._trainX, ABSVR._trainy, params=ABSVR._param)
        else:
            tune.run(ABSVR._trainX, ABSVR._trainy)
        model = tune.model
        ABSVR._param = tune.params
        del tune, model_info
        return model

    def _save_trajectory(self, model, init=False):
        print('Plotting Trajectory Model...')
        XX, YY = np.meshgrid(
            np.linspace(ABSVR._info.data_range[0][0], ABSVR._info.data_range[1][0], num=int(1e2), endpoint=True),
            np.linspace(ABSVR._info.data_range[0][1], ABSVR._info.data_range[1][1], num=int(1e2), endpoint=True))
        Z, V = model.predict(np.c_[XX.ravel(), YY.ravel()], return_var=True)
        L = self.lfxn(Z, V, plot=True)
        Z = Z.reshape(XX.shape)
        V = V.reshape(XX.shape)
        L = L.reshape(XX.shape)

        added_samples = np.loadtxt(r'{}\added.dat'.format(ABSVR._info.root_path))
        if added_samples.ndim == 1:
            added_samples = added_samples.reshape(1, -1)

        # Plot Surface
        fig1 = plt.figure(1)
        ax1 = fig1.gca(projection='3d')
        ax1.plot_surface(XX, YY, Z, cmap=plt.get_cmap('nipy_spectral'))
        ax1.set_xlabel('{}'.format(ABSVR._plot['xlabel']))
        ax1.set_ylabel('{}'.format(ABSVR._plot['ylabel']))
        ax1.set_zlabel('{}'.format(ABSVR._plot['zlabel']))
        if not init:
            ax1.set_title('{} | Added Samples = {}'.format(ABSVR._plot['title'], added_samples.shape[0]))
        else:
            ax1.set_title('{} | Initial '.format(ABSVR._plot['title']))
        ax1.view_init(azim=120)
        plt.savefig('{}/SurfacePlot/{}.png'.format(ABSVR._info.root_path, self.iter))
        plt.close(fig1)
        del fig1, ax1

        # # Plot Mean Contour

        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(111)
        ax2.set_xlabel('{}'.format(ABSVR._plot['xlabel']))
        ax2.set_ylabel('{}'.format(ABSVR._plot['ylabel']))
        if not init:
            ax2.set_title('Mean Zero Contour | Added Samples = {}'.format(len(added_samples)))
        else:
            ax2.set_title('Mean Zero Contour | Initial '.format(ABSVR._plot['title']))

        contour = ax2.contourf(XX, YY, Z, cmap=plt.get_cmap('nipy_spectral'), alpha=0.8)
        cbar = fig2.colorbar(contour)
        cbar.ax.set_ylabel('{}'.format(ABSVR._plot['zlabel']))
        cs = ax2.contour(XX, YY, Z, [0.0], colors='k', linewidths=2)
        temp_: list = [i for i in cs.allsegs[0]]
        dat0 = np.vstack(temp_)
        if len(temp_) == 1:
            ax2.plot(temp_[0][:, 0], temp_[0][:, 1], c='k')
        elif len(temp_) == 2:
            ax2.plot(temp_[0][:, 0], temp_[0][:, 1], temp_[1][:, 0], temp_[1][:, 1], c='k')

        ax2.set_xlim(ABSVR._info.data_range[0][0], ABSVR._info.data_range[1][0])
        ax2.set_ylim(ABSVR._info.data_range[0][1], ABSVR._info.data_range[1][1])
        if not init:
            plt.scatter(added_samples[:, 0], added_samples[:, 1], c='r', s=10, marker='x', label='Added Samples')
        plt.scatter(ABSVR._info.initialX[:, 0], ABSVR._info.initialX[:, 1], c='k', s=10, label='Initial Data')
        ax2.legend(loc=2)
        np.savetxt('{}/ContourPlot/Mean/{}.dat'.format(ABSVR._info.root_path, self.iter), dat0)

        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
        plt.savefig('{}/ContourPlot/Mean/{}.png'.format(ABSVR._info.root_path, self.iter), bbox_inches='tight')
        plt.close(fig2)
        del fig2, ax2

        # # Plot Variance Contour

        fig3 = plt.figure(3)
        ax3 = fig3.add_subplot(111)
        ax3.set_xlabel('{}'.format(ABSVR._plot['xlabel']))
        ax3.set_ylabel('{}'.format(ABSVR._plot['ylabel']))
        if not init:
            ax3.set_title('Variance Contour | Added Samples = {}'.format(len(added_samples)))
        else:
            ax3.set_title('Variance Contour | Initial ')

        contour = ax3.contourf(XX, YY, V, cmap=plt.get_cmap('nipy_spectral'), alpha=0.8)
        cbar = fig3.colorbar(contour)
        ax3.set_xlim(ABSVR._info.data_range[0][0], ABSVR._info.data_range[1][0])
        ax3.set_ylim(ABSVR._info.data_range[0][1], ABSVR._info.data_range[1][1])
        if not init:
            plt.scatter(added_samples[:, 0], added_samples[:, 1], c='r', s=10, marker='x', label='Added Samples')
        plt.scatter(ABSVR._info.initialX[:, 0], ABSVR._info.initialX[:, 1], c='k', s=10, label='Initial Data')
        ax3.legend(loc=2)

        box = ax3.get_position()
        ax3.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
        plt.savefig('{}/ContourPlot/Variance/{}.png'.format(ABSVR._info.root_path, self.iter), bbox_inches='tight')
        plt.close(fig3)
        del fig3, ax3
        del Z, V
        # # Plot Learn Fxn Contour

        fig4 = plt.figure(3)
        ax4 = fig4.add_subplot(111)
        ax4.set_xlabel('{}'.format(ABSVR._plot['xlabel']))
        ax4.set_ylabel('{}'.format(ABSVR._plot['ylabel']))
        if not init:
            ax4.set_title('{} Contour | Added Samples = {}'.format(self._info.learnfxn.upper(), len(added_samples)))
        else:
            ax4.set_title('{} Contour | Initial '.format(self._info.learnfxn.upper()))

        contour = ax4.contourf(XX, YY, L, cmap=plt.get_cmap('nipy_spectral'), alpha=0.8)
        cbar = fig4.colorbar(contour)
        ax4.set_xlim(ABSVR._info.data_range[0][0], ABSVR._info.data_range[1][0])
        ax4.set_ylim(ABSVR._info.data_range[0][1], ABSVR._info.data_range[1][1])
        if not init:
            plt.scatter(added_samples[:, 0], added_samples[:, 1], c='r', s=10, marker='x', label='Added Samples')
        plt.scatter(ABSVR._info.initialX[:, 0], ABSVR._info.initialX[:, 1], c='k', s=10, label='Initial Data')
        ax4.legend(loc=2)

        box = ax4.get_position()
        ax4.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
        plt.savefig('{}/ContourPlot/{}/{}.png'.format(ABSVR._info.root_path, self._info.learnfxn.upper(), self.iter,
                                                      bbox_inches='tight'))
        plt.close(fig4)
        del fig4, ax4

        # # Import Reference Flutter boundary
        rmarques = np.loadtxt(r'D:\PYTHON\.PROJECTS\BayesianSVR\SIMULATIONS\.REFERENCE\Ref_Marques.dat')
        rhyb = np.loadtxt(r'D:\PYTHON\.PROJECTS\BayesianSVR\SIMULATIONS\.REFERENCE\.REF_HYB-N\zero_contour.dat')
        fig5 = plt.figure(5)
        ax5 = fig5.add_subplot(111)
        ax5.set_xlabel('Mach')
        ax5.set_ylabel('Flutter Speed Index')
        ax5.set_title('Flutter Boundary | Beta = {0:.3f}'.format(ABSVR._param['beta']))

        plt.scatter(rhyb[::100, 0], rhyb[::100, 1], c="r", s=15, marker="+", label="Present: Reference Model")
        plt.scatter(rmarques[:, 0], rmarques[:, 1], c="y", s=15, marker="v", label="Marques et al (2020)")

        if len(temp_) == 1:
            ax5.plot(temp_[0][:, 0], temp_[0][:, 1], c='b')
            ax5.plot([], [], c='b', label="Present: {} ({})".format(ABSVR._plot['title'], ABSVR._info.learnfxn.upper()))
        elif len(temp_) == 2:
            ax5.plot(temp_[0][:, 0], temp_[0][:, 1], temp_[1][:, 0], temp_[1][:, 1], c='b')
            ax5.plot([], [], c='b', label="Present: {} ({})".format(ABSVR._plot['title'], ABSVR._info.learnfxn.upper()))

        ax5.set_xlim(ABSVR._info.data_range[0][0], ABSVR._info.data_range[1][0])
        ax5.set_ylim(ABSVR._info.data_range[0][1], ABSVR._info.data_range[1][1])
        ax5.legend()

        plt.savefig('{}/ReferencePlot/{}.png'.format(ABSVR._info.root_path, self.iter), bbox_inches='tight')
        plt.close(fig5)
        del rmarques, rhyb, XX, YY

    def _main_loop(self, model):
        print('-' * 100, '\n', 'Main Loop\n', '-' * 100)
        out_iter = 1
        cov = 1
        iterations = []
        while self.iter <= 100:
            print('-' * 100, '\n', 'Main Loop Iteration: {}\n'.format(out_iter), '-' * 100)
            mcs_samples = self._generate_mcs()
            model = self._inner_loop(mcs_samples, model)
            mean = model.predict(mcs_samples)
            prob_fail = self._probability(mean)
            cov = np.sqrt((1 - prob_fail) / (prob_fail * len(mcs_samples)))
            print('COV Outer: {}'.format(cov))
            iterations.append([out_iter, self.iter])
            out_iter += 1
            self.iter += 1
            if self.iter == 101 or cov < 0.005:
                break
        self._post_process(iterations)
        del mcs_samples
    
    def _inner_loop(self, mcs_samples, model):
        print('-' * 100, '\n', 'Inner Loop\n')
        while self.iter <= 101:
            print('-' * 100, '\n', 'Inner Loop Iteration {}'.format(self.iter))
            in_mean, in_var = model.predict(mcs_samples, return_var=True)
            idx, cond, lvalue = self.lfxn(in_mean, in_var)
            X_new = mcs_samples[idx, :]

            if self.iter % 2 == 0:
                pred = self.rmodel.predict(X_new.reshape(1, -1), model='gpr')
            else:
                pred = self.rmodel.predict(X_new.reshape(1, -1), model='svr')
            new_sample = np.append(X_new, pred.ravel()).reshape(1, -1)
            prob_fail = self._probability(in_mean)
            cov = np.sqrt((1 - prob_fail) / (prob_fail * len(mcs_samples)))

            with open(r'{}\added.dat'.format(ABSVR._info.root_path), 'a+') as f:
                np.savetxt(f, new_sample.reshape(1, -1))
                f.close()
            with open(r'{}\prob.dat'.format(ABSVR._info.root_path), 'a+') as f:
                np.savetxt(f, np.array([int(self.iter), prob_fail]).reshape(1, -1))
                f.close()
            with open(r'{}\cov.dat'.format(ABSVR._info.root_path), 'a+') as f:
                np.savetxt(f, np.array([int(self.iter), cov]).reshape(1, -1))
                f.close()
            with open(r'{}\{}.dat'.format(ABSVR._info.root_path, ABSVR._info.learnfxn), 'a+') as f:
                np.savetxt(f, np.array([int(self.iter), lvalue]).reshape(1, -1))
                f.close()

            print('Added Sample: {}'.format(new_sample))
            print('COV Inner: {}'.format(cov))
            self._save_trajectory(model, init=False)
            ABSVR._trainX = np.vstack((ABSVR._trainX, X_new.reshape(1, -1)))
            ABSVR._trainy = np.append(ABSVR._trainy, pred.ravel())
            new_model = self._generate_model(n_trial=10, reload=True)
            del model
            model = new_model
            if self.iter == 100 or cond == 0:
                break
            self.iter += 1
            del in_mean, in_var, new_sample
        return model
