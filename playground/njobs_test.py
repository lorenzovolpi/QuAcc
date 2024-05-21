import joblib
import numpy as np
import quapy as qp
from quapy.method.aggregative import PACC
from sklearn.linear_model import LogisticRegression

import quacc as qc
from quacc.dataset import DatasetProvider as DP
from quacc.error import f1_macro
from quacc.experiments.util import split_validation
from quacc.models.cont_table import QuAcc1xN2
from quacc.models.model_selection import GridSearchCAP

qp.environ["_R_SEED"] = 0
qp.environ["SAMPLE_SIZE"] = 250

if __name__ == "__main__":
    L, V, U = DP.uci_multiclass("letter")
    V, val_prot = split_validation(V)

    h = LogisticRegression().fit(*L.Xy)

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

    print(qc.env["N_JOBS"])

    joblib.parallel_config(verbose=10)
    model = GridSearchCAP(
        QuAcc1xN2(h, f1_macro, PACC(LogisticRegression())),
        pacc_lr_params,
        val_prot,
        f1_macro,
        refit=True,
    ).fit(V)
