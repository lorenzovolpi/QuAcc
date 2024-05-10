from time import time

import numpy as np
import quapy as qp
from quapy.method.aggregative import EMQ, PACC, KDEyML
from quapy.protocol import UPP
from sklearn.linear_model import LogisticRegression

import quacc.error
from quacc.dataset import DatasetProvider as DP
from quacc.error import f1_macro, vanilla_acc
from quacc.experiments.util import split_validation
from quacc.models.cont_table import QuAcc1xN2, QuAccNxN
from quacc.models.model_selection import GridSearchCAP as GSCAP
from quacc.utils.commons import true_acc

qp.environ["SAMPLE_SIZE"] = 250
NUM_TEST = 1000
qp.environ["_R_SEED"] = 0

pacc_lr_params = {
    "q_class__classifier__C": np.logspace(-3, 3, 7),
    "q_class__classifier__class_weight": [None, "balanced"],
    # "add_X": [True, False],
    "add_posteriors": [True, False],
    "add_y_hat": [True, False],
    "add_maxconf": [True, False],
    "add_negentropy": [True, False],
    "add_maxinfsoft": [True, False],
}
emq_lr_params = pacc_lr_params | {"q_class__recalib": [None, "bcts"]}
kde_lr_params = pacc_lr_params | {"q_class__bandwidth": np.linspace(0.01, 0.2, 20)}

if __name__ == "__main__":
    dataset_name = "C18"
    L, V, U = DP.rcv1_multiclass(dataset_name)
    V, val_prot = split_validation(V)
    h = LogisticRegression()
    test_prot = UPP(U, repeats=NUM_TEST, return_type="labelled_collection", random_state=0)
    accs = [("vanilla_acc", vanilla_acc), ("f1", f1_macro)]

    h.fit(*L.Xy)
    results = []
    for acc_name, acc_fn in accs:
        quants = [
            ("EMQ", EMQ(LogisticRegression()), emq_lr_params),
            ("PACC", PACC(LogisticRegression()), pacc_lr_params),
            ("KDEy", KDEyML(LogisticRegression()), kde_lr_params),
        ]
        for q_name, q, params in quants[2:]:
            methods = [
                (
                    f"QuAcc({q_name})1xn2-OPT-norefit",
                    GSCAP(QuAcc1xN2(h, acc_fn, q), params, val_prot, acc_fn, raise_errors=True, refit=False),
                ),
                (
                    f"QuAcc({q_name})nxn-OPT-norefit",
                    GSCAP(QuAccNxN(h, acc_fn, q), params, val_prot, acc_fn, raise_errors=True, refit=False),
                ),
                (
                    "QuAcc(KDEy)1xn2-OPT",
                    GSCAP(QuAcc1xN2(h, acc_fn, q), params, val_prot, acc_fn, raise_errors=True),
                ),
            ]
            for method_name, method in methods[2:]:
                t_init = time()
                method.fit(V)
                true_accs = [true_acc(h, acc_fn, Ui) for Ui in test_prot()]
                estim_accs = [method.predict(Ui.X) for Ui in test_prot()]
                mae = quacc.error.mae(true_accs, estim_accs)
                t_method = time() - t_init
                results.append((method_name, acc_name, mae, t_method))
    for method_name, acc_name, mae, t_method in results:
        print(f"{method_name} on {acc_name}: {mae=} [took {t_method:.3f}s]")
