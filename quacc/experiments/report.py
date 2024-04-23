import itertools
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import quapy as qp

from quacc.utils.commons import get_results_path, load_json_file, save_json_file


def _get_shift(test_prevs: np.ndarray, train_prev: np.ndarray | float, decimals=2):
    """
    Computes the shift of an array of prevalence values for a set of test sample in
    relation to the prevalence value of the training set.

    :param test_prevs: prevalence values for the test samples
    :param train_prev: prevalence value for the training set
    :param decimals: rounding decimals for the result (default=2)
    :return: an ndarray with the shifts for each test sample, shaped as (n,1) (ndim=2)
    """
    if test_prevs.ndim == 1:
        test_prevs = test_prevs[:, np.newaxis]
    train_prevs = np.tile(train_prev, (test_prevs.shape[0], 1))
    # _shift = nae(test_prevs, train_prevs)
    _shift = qp.error.ae(test_prevs, train_prevs)
    return np.around(_shift, decimals=decimals)


class TestReport:
    def __init__(
        self,
        basedir,
        cls_name,
        acc_name,
        dataset_name,
        train_prev,
        val_prev,
        method_name,
    ):
        self.basedir = basedir
        self.cls_name = cls_name
        self.acc_name = acc_name
        self.dataset_name = dataset_name
        self.train_prev = train_prev
        self.val_prev = val_prev
        self.method_name = method_name

    def get_path(self):
        return get_results_path(self.basedir, self.cls_name, self.acc_name, self.dataset_name, self.method_name)

    def add_result(self, test_prevs, true_accs, estim_accs, t_train, t_test_ave):
        self.test_prevs = test_prevs
        self.estim_accs = estim_accs
        self.true_accs = true_accs
        self.t_train = t_train
        self.t_test_ave = t_test_ave

        return self

    def save_json(self, basedir, acc_name):
        result = {
            "basedir": self.basedir,
            "cls_name": self.cls_name,
            "acc_name": self.acc_name,
            "dataset_name": self.dataset_name,
            "train_prev": self.train_prev,
            "val_prev": self.val_prev,
            "method_name": self.method_name,
            "test_prevs": self.test_prevs,
            "true_accs": self.true_accs,
            "estim_accs": self.estim_accs,
            "t_train": self.t_train,
            "t_test_ave": self.t_test_ave,
        }

        save_json_file(self.get_path(), result)

    @classmethod
    def load_json(cls, path) -> "TestReport":
        def _test_report_hook(_dict):
            return TestReport(
                basedir=_dict["basedir"],
                cls_name=_dict["cls_name"],
                acc_name=_dict["acc_name"],
                dataset_name=_dict["dataset_name"],
                train_prev=_dict["train_prev"],
                val_prev=_dict["val_prev"],
                method_name=_dict["method_name"],
            ).add_result(
                test_prevs=_dict["test_prevs"],
                true_accs=_dict["true_accs"],
                estim_accs=_dict["estim_accs"],
                t_train=_dict["t_train"],
                t_test_ave=_dict["t_test_ave"],
            )

        return load_json_file(path, object_hook=_test_report_hook)


class Report:
    def __init__(self, results: dict[str, list[TestReport]]):
        self.results = results

    @classmethod
    def load_results(cls, basedir, cls_name, acc_name, dataset_name="*", method_name="*") -> "Report":
        _results = defaultdict(lambda: [])
        if isinstance(method_name, str):
            method_name = [method_name]
        if isinstance(dataset_name, str):
            dataset_name = [dataset_name]
        for dataset_, method_ in itertools.product(dataset_name, method_name):
            path = get_results_path(basedir, cls_name, acc_name, dataset_, method_)
            for file in glob(path):
                if file.endswith(".json"):
                    method = Path(file).stem
                    _res = TestReport.load_json(file)
                    _results[method].append(_res)
        return Report(_results)

    def filter_by_method(self, methods=None):
        if methods is None:
            return Report(self.results)

        if isinstance(methods, str):
            methods = [methods]
        return Report({_m: _rs for _m, _rs in self.results.items() if _m in methods})

    def diagonal_plot_data(self):
        dfs = []
        for _method, _results in self.results.items():
            _true_acc = np.hstack([_r.true_accs for _r in _results])
            _estim_acc = np.hstack([_r.estim_accs for _r in _results])
            method_df = pd.DataFrame(np.vstack([_true_acc, _estim_acc]).T, columns=["true_accs", "estim_accs"])
            method_df.loc[:, "method"] = np.tile(_method, (len(method_df),))
            dfs.append(method_df)

        return pd.concat(dfs, axis=0, ignore_index=True)

    # def delta_plot_data(self):
    #     dfs = []
    #     for _method, _results in self.results.items():
    #         if isinstance(_results[0].test_prevs[0], float):
    #             _prev = np.hstack([_r.test_prevs for _r in _results])
    #         else:
    #             _prev = np.vstack([_r.test_prevs for _r in _results])
    #             _prev = np.fromiter((tuple(tp) for tp in _prev), dtype="object")

    #         _true_accs = np.hstack([_r.true_accs for _r in _results])
    #         _estim_accs = np.hstack([_r.estim_accs for _r in _results])
    #         _acc_err = np.abs(_true_accs - _estim_accs)
    #         method_df = pd.DataFrame(
    #             np.vstack([_prev, _acc_err]).T, columns=["prevs", "acc_err"]
    #         )
    #         method_df.loc[:, "method"] = np.tile(_method, (len(method_df),))
    #         dfs.append(method_df.sort_values(by="prevs"))

    #     return pd.concat(dfs, axis=0, ignore_index=True)

    def shift_plot_data(self):
        dfs = []
        for _method, _results in self.results.items():
            _shift = np.hstack([_get_shift(np.array(_r.test_prevs), _r.train_prev) for _r in _results])

            _true_accs = np.hstack([_r.true_accs for _r in _results])
            _estim_accs = np.hstack([_r.estim_accs for _r in _results])
            _acc_err = np.abs(_true_accs - _estim_accs)
            method_df = pd.DataFrame(np.vstack([_shift, _acc_err]).T, columns=["shifts", "acc_err"])
            method_df.loc[:, "method"] = np.tile(_method, (len(method_df),))
            dfs.append(method_df.sort_values(by="shifts"))

        return pd.concat(dfs, axis=0, ignore_index=True)
