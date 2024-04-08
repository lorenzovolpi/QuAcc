import itertools
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from quacc.error import nae
from quacc.utils.commons import get_results_path, load_json_file, save_json_file


def _get_shift(index: np.ndarray, train_prev: np.ndarray):
    index = np.array([np.array(tp) for tp in index])
    train_prevs = np.tile(train_prev, (index.shape[0], 1))
    _shift = nae(index, train_prevs)
    return np.around(_shift, decimals=2)


class TestReport:
    def __init__(
        self,
        basedir,
        cls_name,
        acc_name,
        dataset_name,
        method_name,
        train_prev,
        val_prev,
    ):
        self.basedir = basedir
        self.cls_name = cls_name
        self.acc_name = acc_name
        self.dataset_name = dataset_name
        self.method_name = method_name
        self.train_prev = train_prev
        self.val_prev = val_prev

    @property
    def path(self):
        return get_results_path(
            self.basedir,
            self.cls_name,
            self.acc_name,
            self.dataset_name,
            self.method_name,
        )

    def add_result(self, test_prevs, true_accs, estim_accs, t_train, t_test_ave):
        self.test_prevs = test_prevs
        self.true_accs = true_accs
        self.estim_accs = estim_accs
        self.t_train = t_train
        self.t_test_ave = t_test_ave
        return self

    def save_json(self, basedir):
        if not all([hasattr(self, _attr) for _attr in ["true_accs", "estim_accs"]]):
            raise AttributeError("Incomplete report cannot be dumped")

        result = {
            "basedir": self.basedir,
            "cls_name": self.cls_name,
            "acc_name": self.acc_name,
            "dataset_name": self.dataset_name,
            "method_name": self.method_name,
            "train_prev": self.train_prev,
            "val_prev": self.val_prev,
            "test_prevs": self.test_prevs,
            "true_accs": self.true_accs,
            "estim_accs": self.estim_accs,
            "t_train": self.t_train,
            "t_test_ave": self.t_test_ave,
        }

        save_json_file(self.path, result)

    @classmethod
    def load_json(cls, path) -> "TestReport":
        def _test_report_hook(_dict):
            return TestReport(
                basedir=_dict["basedir"],
                cls_name=_dict["cls_name"],
                acc_name=_dict["acc_name"],
                dataset_name=_dict["dataset_name"],
                method_name=_dict["method_name"],
                train_prev=_dict["train_prev"],
                val_prev=_dict["val_prev"],
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
    def load_results(
        cls, basedir, cls_name, acc_name, dataset_name="*", method_name="*"
    ) -> "Report":
        _results = defaultdict(lambda: [])
        if isinstance(method_name, str):
            method_name = [method_name]
        if isinstance(dataset_name, str):
            dataset_name = [dataset_name]
        for dataset_, method_ in itertools.product(dataset_name, method_name):
            path = get_results_path(basedir, cls_name, acc_name, dataset_, method_)
            for file in glob(path):
                if file.endswith(".json"):
                    # print(file)
                    method = Path(file).stem
                    _res = TestReport.load_json(file)
                    _results[method].append(_res)
        return Report(_results)

    def train_table(self):
        pass

    def test_table(self):
        pass

    def shift_table(self):
        pass

    def diagonal_plot_data(self):
        methods = []
        true_accs = []
        estim_accs = []
        for _method, _results in self.results.items():
            methods.append(_method)
            _true_acc = np.array([_r.true_accs for _r in _results]).flatten()
            _estim_acc = np.array([_r.estim_accs for _r in _results]).flatten()
            true_accs.append(_true_acc)
            estim_accs.append(_estim_acc)

        return methods, true_accs, estim_accs

    def delta_plot_data(self, stdev=False):
        methods = []
        prevs = []
        acc_errs = []
        stdevs = None if stdev is None else []
        for _method, _results in self.results.items():
            methods.append(_method)
            _prevs = np.array(
                [_r.test_prevs for _r in _results]
            ).flatten()  # should not be flattened, check this
            _true_accs = np.array([_r.true_accs for _r in _results]).flatten()
            _estim_accs = np.array([_r.estim_accs for _r in _results]).flatten()
            _acc_errs = np.abs(_true_accs - _estim_accs)
            df = pd.DataFrame(
                np.array([_prevs, _acc_errs]).T, columns=["prevs", "errs"]
            )
            df_acc_errs = df.groupby(["prevs"]).mean().reset_index()
            prevs.append(df_acc_errs["prevs"].to_numpy())
            acc_errs.append(df_acc_errs["errs"].to_numpy())
            if stdev:
                df_stdevs = df.groupby(["prevs"]).std().reset_index()
                stdevs.append(df_stdevs["errs"].to_numpy())

        return methods, prevs, acc_errs, stdevs

    def shift_plot_data(self):
        pass
