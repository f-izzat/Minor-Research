from utilities import *


class Hybrid:
    def __init__(self, svr_path: str, gpr_path: str):
        # # Load GPR Reference Model
        self.gmodel = jload(open('{}/Model.front'.format(gpr_path), 'rb'))
        self.gmodel.reload()
        # # Load Epsilon-SVR Reference Model
        self.svrmodel = jload(open('{}/Model.front'.format(svr_path), 'rb'))

    def predict(self, testX: array, model: str, return_var: bool = False):
        if model == 'gpr':
            if return_var:
                return self.gmodel.predict(testX, return_var=True)
            else:
                return self.gmodel.predict(testX)
        else:
            return self.svrmodel.predict(testX)
