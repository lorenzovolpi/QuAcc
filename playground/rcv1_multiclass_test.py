import numpy as np
import quapy as qp
from quapy.method.aggregative import EMQ
from quapy.protocol import UPP
from sklearn.linear_model import LogisticRegression

import quacc.error
from quacc.dataset import DatasetProvider as DP
from quacc.error import vanilla_acc
from quacc.experiments.util import split_validation
from quacc.models.cont_table import QuAcc1xN2
from quacc.models.model_selection import GridSearchCAP as GSCAP
from quacc.utils.commons import true_acc

qp.environ["SAMPLE_SIZE"] = 250
NUM_TEST = 1000
qp.environ["_R_SEED"] = 0

np.seterr(all="raise")

emq_lr_params = {
    "q_class__classifier__C": np.logspace(-3, 3, 7),
    "q_class__classifier__class_weight": [None, "balanced"],
    "q_class__recalib": [None, "bcts"],
    # "add_X": [True, False],
    "add_posteriors": [True, False],
    "add_y_hat": [True, False],
    "add_maxconf": [True, False],
    "add_negentropy": [True, False],
    "add_maxinfsoft": [True, False],
}

if __name__ == "__main__":
    dataset_name = "C18"
    L, V, U = DP.rcv1_multiclass(dataset_name)
    V, val_prot = split_validation(V)
    h = LogisticRegression()
    test_prot = UPP(U, repeats=NUM_TEST, return_type="labelled_collection", random_state=0)
    acc_name, acc_fn = "vanilla_acc", vanilla_acc

    h.fit(*L.Xy)
    q = EMQ(LogisticRegression())
    cap = QuAcc1xN2(h, acc_fn, q)
    method_name, method = (
        "QuAcc(EMQ)1xn2-OPT",
        GSCAP(cap, emq_lr_params, val_prot, acc_fn, n_jobs=1, raise_errors=True),
    )
    method.fit(V)
    true_accs = [true_acc(h, acc_fn, Ui) for Ui in test_prot()]
    estim_accs = [method.predict(Ui.X) for Ui in test_prot()]
    mae = quacc.error.mae(true_accs, estim_accs)
    print(f"{method_name} on {acc_name}: {mae=}")
