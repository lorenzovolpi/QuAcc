import numpy as np

from quacc.experiments.generators import get_method_names
from quacc.experiments.report import Report
from quacc.plot.matplotlib import plot_delta, plot_diagonal, plot_shift


def save_plot_diagonal(
    basedir, cls_name, acc_name, dataset_name="*", report: Report = None
):
    methods = get_method_names()
    report = (
        Report.load_results(
            basedir,
            cls_name,
            acc_name,
            dataset_name=dataset_name,
            method_name=methods,
        )
        if report is None
        else report
    )
    _methods, _true_accs, _estim_accs = report.diagonal_plot_data()
    plot_diagonal(
        method_names=_methods,
        true_accs=_true_accs,
        estim_accs=_estim_accs,
        cls_name=cls_name,
        acc_name=acc_name,
        dataset_name=dataset_name,
        basedir=basedir,
    )


def save_plot_delta(
    basedir, cls_name, acc_name, dataset_name="*", stdev=False, report: Report = None
):
    methods = get_method_names()
    report = (
        Report.load_results(
            basedir,
            cls_name,
            acc_name,
            dataset_name=dataset_name,
            method_name=methods,
        )
        if report is None
        else report
    )
    _methods, _prevs, _acc_errs, _stdevs = report.delta_plot_data(stdev=stdev)
    plot_delta(
        method_names=_methods,
        prevs=_prevs,
        acc_errs=_acc_errs,
        cls_name=cls_name,
        acc_name=acc_name,
        dataset_name=dataset_name,
        basedir=basedir,
        stdevs=_stdevs,
    )


def save_plot_shift(
    basedir, cls_name, acc_name, dataset_name="*", report: Report = None
):
    methods = get_method_names()
    report = (
        Report.load_results(
            basedir,
            cls_name,
            acc_name,
            dataset_name=dataset_name,
            method_name=methods,
        )
        if report is None
        else report
    )
    _methods, _shifts, _acc_errs = report.shift_plot_data()
    plot_shift(
        method_names=_methods,
        prevs=_shifts,
        acc_errs=_acc_errs,
        cls_name=cls_name,
        acc_name=acc_name,
        dataset_name=dataset_name,
        basedir=basedir,
    )
