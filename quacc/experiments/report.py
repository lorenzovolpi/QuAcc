import itertools
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from quacc.error import nae
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
    _shift = nae(test_prevs, train_prevs)
    return np.around(_shift, decimals=decimals)[:, np.newaxis]


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
            _prev = [np.array(_r.test_prevs) for _r in _results]
            # if prevalence values are floats, transform them in (1,) arrays
            prev_ndim = _prev[0].ndim
            if prev_ndim == 1:
                _prev = [rp[:, np.newaxis] for rp in _prev]
            # join all prevalence values in a single array
            _prev = np.vstack(_prev)
            # join all true_accs values in a single array
            _true_accs = np.hstack([_r.true_accs for _r in _results])
            # join all estim_accs values in a single array
            _estim_accs = np.hstack([_r.estim_accs for _r in _results])
            # compute the absolute earror for each prevalence value
            _acc_err = np.abs(_true_accs - _estim_accs)[:, np.newaxis]
            # build a df with prevs and errors
            df = pd.DataFrame(np.hstack([_prev, _acc_err]))
            # build a df by grouping by the first n-1 columns and compute the mean
            df_mean = df.groupby(df.columns[:-1].to_list()).mean().reset_index()
            # insert unique prevs in the "prevs" list
            if prev_ndim == 1:
                prevs.append(df_mean.iloc[:, :-1].to_numpy())
            else:
                prevs.append(
                    np.fromiter(
                        (tuple(p) for p in df_mean.iloc[:, :-1].to_numpy()),
                        dtype="object",
                    )
                )
            # insert the errors in the right array
            acc_errs.append(df_mean.iloc[:, -1].to_numpy())
            # if stdev is required repeat last steps for std()
            if stdev:
                df_std = df.groupby(df.columns[:-1].to_list()).std().reset_index()
                stdevs.append(df_std.iloc[:, -1].to_numpy())

        return methods, prevs, acc_errs, stdevs

    def shift_plot_data(self):
        methods = []
        shifts = []
        acc_errs = []
        for _method, _results in self.results.items():
            methods.append(_method)
            _test_prev = [np.array(_r.test_prevs) for _r in _results]
            _train_prev = [_r.train_prev for _r in _results]
            # if prevalence values are floats, transform them in (1,) arrays
            if _test_prev[0].ndim == 1:
                _test_prev = [rp[:, np.newaxis] for rp in _test_prev]
            # join values in a single array per type
            _true_accs = np.hstack([_r.true_accs for _r in _results])
            _estim_accs = np.hstack([_r.estim_accs for _r in _results])
            # compute the shift for each test sample
            _shift = (
                np.vstack(
                    [_get_shift(p, tp) for (p, tp) in zip(_test_prev, _train_prev)]
                ),
            )
            # compute the absolute earror for each prevalence value
            _acc_err = np.abs(_true_accs - _estim_accs)[:, np.newaxis]
            # build a df with prevs and errors
            df = pd.DataFrame(np.hstack([_shift, _acc_err]))
            # build a df by grouping by the first n-1 columns and compute the mean
            df_mean = df.groupby(df.columns[:-1].to_list()).mean().reset_index()
            # insert unique prevs in the "prevs" list
            shifts.append(df_mean.iloc[:, :-1].to_numpy())
            # insert the errors in the right array
            acc_errs.append(df_mean.iloc[:, -1].to_numpy())

        return methods, shifts, acc_errs
