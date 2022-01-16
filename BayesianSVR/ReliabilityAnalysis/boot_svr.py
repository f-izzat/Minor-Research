from utilities import *
from dataclasses import dataclass as _dataclass
from surrogate_models.svr import Tune, SVRInfo


@_dataclass(frozen=True)
class BootInfo:
    svrtype: str
    rmodel: Any
    root_path: str
    data_range: list
    initialX: array
    initialy: array

    max_iter: int = 100
    n_boot: int = 10
    n_mcs: int = int(1e4)


class BootSVR:
    _info: BootInfo
    _space: dict = {'lb': {'C': 1, 'theta': 0.001, 'eps': 0.00001},
                    'ub': {'C': 1000, 'theta': 10, 'eps': 0.01},
                    'x0': {'C': 10, 'theta': 1, 'eps': 0.001},
                    'kernel_type': 'GAUSSIAN',
                    'ARD': 1}
    _bestpar: dict
    _plot: dict
    _bcnt: array

    def __init__(self, _info: BootInfo) -> None:
        BootSVR._info = _info
        BootSVR._plot = {'xlabel': 'Mach', 'ylabel': 'Flutter Speed Index', 'zlabel': 'Damping Coefficient',
                         'title': 'BootSVR'}

        # Updated During Training
        self._trainX: array = _info.initialX.copy()
        self._trainy: array = _info.initialy.copy()
        self.iter: int = 0

    @staticmethod
    def _probability(X: array) -> float:
        n_fail = np.sum(X <= 0)
        return n_fail.item() / len(X)

    @staticmethod
    def _generate_mcs():
        print('Generating Monte-Carlo Simulations')
        n_ftr = BootSVR._info.initialX.shape[1]
        mcs = np.random.rand(BootSVR._info.n_mcs, n_ftr)
        min_, max_ = BootSVR._info.data_range[0], BootSVR._info.data_range[1]
        for i in range(mcs.shape[1]):
            mcs[:, i] = (max_[i] - min_[i]) * ((mcs[:, i] - min(mcs[:, i])) / (max(mcs[:, i]) - min(mcs[:, i]))) + min_[
                i]
        return mcs

    @staticmethod
    def _generate_folder():
        # # Create Files for storage
        tmp = np.array([])
        if not exists(r'{}\added.dat'.format(BootSVR._info.root_path)):
            np.savetxt(r'{}\added.dat'.format(BootSVR._info.root_path), tmp)
        else:
            remove(r'{}\added.dat'.format(BootSVR._info.root_path))
            np.savetxt(r'{}\added.dat'.format(BootSVR._info.root_path), tmp)

        if not exists(r'{}\cov.dat'.format(BootSVR._info.root_path)):
            np.savetxt(r'{}\cov.dat'.format(BootSVR._info.root_path), tmp)
        else:
            remove(r'{}\cov.dat'.format(BootSVR._info.root_path))
            np.savetxt(r'{}\cov.dat'.format(BootSVR._info.root_path), tmp)

        if not exists(r'{}\prob.dat'.format(BootSVR._info.root_path)):
            np.savetxt(r'{}\prob.dat'.format(BootSVR._info.root_path), tmp)
        else:
            remove(r'{}\prob.dat'.format(BootSVR._info.root_path))
            np.savetxt(r'{}\prob.dat'.format(BootSVR._info.root_path), tmp)

        if not exists(r'{}\sample_count.dat'.format(BootSVR._info.root_path)):
            np.savetxt(r'{}\sample_count.dat'.format(BootSVR._info.root_path), tmp)
        else:
            remove(r'{}\sample_count.dat'.format(BootSVR._info.root_path))
            np.savetxt(r'{}\sample_count.dat'.format(BootSVR._info.root_path), tmp)

        if not exists(r'{}\ContourPlot'.format(BootSVR._info.root_path)):
            makedirs(r'{}\ContourPlot'.format(BootSVR._info.root_path))
        else:
            rmtree(r'{}\ContourPlot'.format(BootSVR._info.root_path))
            makedirs(r'{}\ContourPlot'.format(BootSVR._info.root_path))

        if not exists(r'{}\ReferencePlot'.format(BootSVR._info.root_path)):
            makedirs(r'{}\ReferencePlot'.format(BootSVR._info.root_path))
        else:
            rmtree(r'{}\ReferencePlot'.format(BootSVR._info.root_path))
            makedirs(r'{}\ReferencePlot'.format(BootSVR._info.root_path))

        if not exists(r'{}\SurfacePlot'.format(BootSVR._info.root_path)):
            makedirs(r'{}\SurfacePlot'.format(BootSVR._info.root_path))
        else:
            rmtree(r'{}\SurfacePlot'.format(BootSVR._info.root_path))
            makedirs(r'{}\SurfacePlot'.format(BootSVR._info.root_path))

    def _save_trajectory(self, model, init=False):
        print('Plotting Trajectory Model...')
        XX, YY = np.meshgrid(
            np.linspace(BootSVR._info.data_range[0][0], BootSVR._info.data_range[1][0], num=int(1e3), endpoint=True),
            np.linspace(BootSVR._info.data_range[0][1], BootSVR._info.data_range[1][1], num=int(1e3), endpoint=True))

        Z = model.predict(np.c_[XX.ravel(), YY.ravel()])
        Z = Z.reshape(XX.shape)

        added_samples = np.loadtxt(r'{}\added.dat'.format(BootSVR._info.root_path))
        if added_samples.ndim == 1:
            added_samples = added_samples.reshape(1, -1)

        # Plot Surface
        fig1 = plt.figure(1)
        ax1 = fig1.gca(projection='3d')
        ax1.plot_surface(XX, YY, Z, cmap=plt.get_cmap('nipy_spectral'))
        ax1.set_xlabel('{}'.format(BootSVR._plot['xlabel']))
        ax1.set_ylabel('{}'.format(BootSVR._plot['ylabel']))
        ax1.set_zlabel('{}'.format(BootSVR._plot['zlabel']))
        if not init:
            ax1.set_title('{} | Added Samples = {}'.format(BootSVR._plot['title'], added_samples.shape[0]))
        else:
            ax1.set_title('{} | Initial '.format(BootSVR._plot['title']))
        ax1.view_init(azim=120)
        plt.savefig('{}/SurfacePlot/{}.png'.format(BootSVR._info.root_path, self.iter))
        plt.close(fig1)

        # # Plot Contour

        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(111)
        ax2.set_xlabel('{}'.format(BootSVR._plot['xlabel']))
        ax2.set_ylabel('{}'.format(BootSVR._plot['ylabel']))
        if not init:
            ax2.set_title('{} Zero Contour | Added Samples = {}'.format(BootSVR._plot['title'], len(added_samples)))
        else:
            ax2.set_title('{} Model | Initial '.format(BootSVR._plot['title']))

        contour = ax2.contourf(XX, YY, Z, cmap=plt.get_cmap('nipy_spectral'), alpha=0.8)
        cbar = fig2.colorbar(contour)
        cbar.ax.set_ylabel('{}'.format(BootSVR._plot['zlabel']))
        cs = ax2.contour(XX, YY, Z, [0.0], colors='k', linewidths=2)
        temp_: list = [i for i in cs.allsegs[0]]
        dat0 = np.vstack(temp_)
        if len(temp_) == 1:
            ax2.plot(temp_[0][:, 0], temp_[0][:, 1], c='k')
        elif len(temp_) == 2:
            ax2.plot(temp_[0][:, 0], temp_[0][:, 1], temp_[1][:, 0], temp_[1][:, 1], c='k')

        ax2.set_xlim(BootSVR._info.data_range[0][0], BootSVR._info.data_range[1][0])
        ax2.set_ylim(BootSVR._info.data_range[0][1], BootSVR._info.data_range[1][1])
        if not init:
            plt.scatter(added_samples[:, 0], added_samples[:, 1], c='r', s=10, marker='x', label='Added Samples')
        plt.scatter(BootSVR._info.initialX[:, 0], BootSVR._info.initialX[:, 1], c='k', s=10, label='Initial Data')
        ax2.legend(loc=2)
        np.savetxt('{}/ContourPlot/{}.dat'.format(BootSVR._info.root_path, self.iter), dat0)

        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)
        plt.savefig('{}/ContourPlot/{}.png'.format(BootSVR._info.root_path, self.iter), bbox_inches='tight')
        plt.close(fig2)

        # # Import Reference Flutter boundary
        rmarques = np.loadtxt(r'D:\PYTHON\.PROJECTS\BayesianSVR\SIMULATIONS\.REFERENCE\Ref_Marques.dat')
        rgpr = np.loadtxt(r'D:\PYTHON\.PROJECTS\BayesianSVR\SIMULATIONS\.REFERENCE\.REF_HYB-N\zero_contour.dat')
        fig3 = plt.figure(3)
        ax3 = fig3.add_subplot(111)
        ax3.set_xlabel('Mach')
        ax3.set_ylabel('Flutter Speed Index')
        ax3.set_title('Flutter Boundary')
        plt.scatter(rgpr[::100, 0], rgpr[::100, 1], c="r", s=12, marker="+", label="Present: Reference")
        plt.scatter(rmarques[:, 0], rmarques[:, 1], c="y", s=12, marker="v", label="[R] Marques et al (2020)")
        if len(temp_) == 1:
            ax3.plot(temp_[0][:, 0], temp_[0][:, 1], c='b')
            ax3.plot([], [], c='b', label="Present: {}".format(BootSVR._plot['title']))
        elif len(temp_) == 2:
            ax3.plot(temp_[0][:, 0], temp_[0][:, 1], temp_[1][:, 0], temp_[1][:, 1], c='b')
            ax3.plot([], [], c='b', label="Present: {}".format(BootSVR._plot['title']))

        ax3.set_xlim(BootSVR._info.data_range[0][0], BootSVR._info.data_range[1][0])
        ax3.set_ylim(BootSVR._info.data_range[0][1], BootSVR._info.data_range[1][1])
        ax3.legend()

        plt.savefig('{}/ReferencePlot/{}.png'.format(BootSVR._info.root_path, self.iter), bbox_inches='tight')
        plt.close(fig3)
        del rmarques, rgpr

    def _learn(self, pred_cond: array, mcs_samples: array):
        print('Learning...')
        """
        Add New Sample
        Positive - Safe, Negative - Fail
        np.count_nonzero: True = Fail, False = Safe
        To predict with reference model scale data
        """
        # # Add Sample Routine
        z = lambda x1: np.abs(((BootSVR._info.n_boot - x1) - x1) / (x1 + (BootSVR._info.n_boot - x1)))
        U = np.array([z(np.count_nonzero(pred_cond[:, i])) for i in range(pred_cond.shape[1])])
        sample = mcs_samples[np.argmin(U), :].reshape(1, -1)
        mcs_samples = np.delete(mcs_samples, np.argmin(U), axis=0)
        if self.iter % 2 == 0:
            pred = BootSVR._info.rmodel.predict(sample, model='gpr')
        else:
            pred = BootSVR._info.rmodel.predict(sample, model='svr')
        self._trainX = np.vstack((self._trainX, sample))
        self._trainy = np.append(self._trainy, pred)
        print('Added Sample {}: {}'.format(self.iter, np.append(sample, pred)))
        with open(r'{}\added.dat'.format(BootSVR._info.root_path), 'a+') as f:
            np.savetxt(f, np.append(sample, pred).reshape(1, -1))
            f.close()
        # # Only take unqiue Training data
        data = np.hstack((self._trainX, self._trainy.reshape(-1, 1)))
        data = np.unique(data, axis=0)
        self._trainX = data[:, :-1]
        self._trainy = data[:, -1]
        print('Number of Training Samples = {}'.format(len(self._trainX)))
        del z, U, pred, sample

        # # Retrain Surrogate
        print('Training Surrogate...')
        svr_info = SVRInfo(svrtype=self._info.svrtype, trainX=self._trainX.copy(), trainy=self._trainy.copy(),
                           search_space=BootSVR._space,
                           optmethod=1, lossfxn=2, approxcv=False,
                           n_fold=len(self._trainX), n_trial=50)
        Tune(svr_info).run(log=True, params=BootSVR._bestpar)
        BootSVR._bestpar = svr_info.Param
        model = svr_info.Model
        print('Current Model CVE = {}\n'.format(svr_info.CVE))
        self._save_trajectory(model, init=False)
        return model, mcs_samples

    def solve(self) -> None:
        self._generate_folder()
        svr_info = SVRInfo(svrtype=self._info.svrtype, trainX=self._trainX.copy(), trainy=self._trainy.copy(),
                           search_space=BootSVR._space,
                           optmethod=1, lossfxn=2, approxcv=False,
                           n_fold=len(self._trainX), n_trial=200)
        Tune(svr_info).run(log=True)
        model = svr_info.Model
        BootSVR._bestpar = svr_info.Param
        self._save_trajectory(model, init=True)
        print('\n\nInitial CVE: {}'.format(svr_info.CVE))

        """ Generate Bootstrap, Loop over each strap and predict at MCS """
        mcs_samples = self._generate_mcs()
        while self.iter < BootSVR._info.max_iter + 1:
            train_data = np.hstack((self._trainX, self._trainy.reshape(-1, 1)))
            print('\nIteration = {}'.format(self.iter))
            # # Bootstrap Loop
            # mcs_samples = self._generate_mcs()
            scount_ = np.zeros((len(train_data), 2))
            scount_[:, 0] = np.arange(len(train_data))
            prob = []
            for i in range(BootSVR._info.n_boot):
                rand = np.unique(np.random.randint(0, len(train_data), int(len(train_data))))
                for j in rand:
                    scount_[int(j), 1] += 1
                np.random.shuffle(rand)
                data = train_data[rand, :]
                # model.train(data[:, :-1], data[:, -1])
                model.train(data[:, :-1], data[:, -1])
                pred = model.predict(mcs_samples)
                prob.append(self._probability(data[:, -1]))
                tmp = pred.ravel() < 0
                if i == 0:
                    np.savetxt('{}\cond.dat'.format(BootSVR._info.root_path), tmp.reshape(1, -1))
                else:
                    with open(r'{}\cond.dat'.format(BootSVR._info.root_path), 'a+') as f:
                        np.savetxt(f, tmp.reshape(1, -1))
                        f.close()
                del tmp
                print('Bootstrap {:4} | Probability = {:.5f} |'.format(i + 1, prob[i]))

            with open(r'{}\sample_count.dat'.format(BootSVR._info.root_path), 'a+') as f:
                f.write('{}'.format(int(self.iter)))
                np.savetxt(f, scount_)
                f.close()

            """ Calculate Convergence """
            current: float = self._probability(train_data[:, -1])
            conv: float = (max(prob) - min(prob)) / current
            with open(r'{}\prob.dat'.format(BootSVR._info.root_path), 'a+') as f:
                f.write('{}, {}'.format(int(self.iter), current))
                f.close()
            with open(r'{}\cov.dat'.format(BootSVR._info.root_path), 'a+') as f:
                f.write('{}, {}'.format(int(self.iter), conv))
                f.close()
            print('CoV: {}'.format(conv))
            print('Current Pf : {}'.format(current))

            """" Learn New Sample """
            del model, train_data
            pred_cond = np.loadtxt(r'{}\cond.dat'.format(BootSVR._info.root_path))
            model, mcs_samples = self._learn(pred_cond, mcs_samples)
            # del mcs_samples
            self.iter += 1
            print('-x-' * 50)


