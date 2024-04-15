import numpy as np

from quacc.experiments.generators import get_method_names
from quacc.experiments.report import Report
from quacc.plot.seaborn import plot_diagonal, plot_shift


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
    df = report.diagonal_plot_data()
    plot_diagonal(
        df=df,
        cls_name=cls_name,
        acc_name=acc_name,
        dataset_name=dataset_name,
        basedir=basedir,
    )


# def save_plot_delta(
#     basedir, cls_name, acc_name, dataset_name="*", stdev=False, report: Report = None
# ):
#     methods = get_method_names()
#     report = (
#         Report.load_results(
#             basedir,
#             cls_name,
#             acc_name,
#             dataset_name=dataset_name,
#             method_name=methods,
#         )
#         if report is None
#         else report
#     )
#     df = report.delta_plot_data()
#     plot_delta(
#         df=df,
#         cls_name=cls_name,
#         acc_name=acc_name,
#         dataset_name=dataset_name,
#         basedir=basedir,
#         stdev=stdev,
#     )


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
    df = report.shift_plot_data()
    plot_shift(
        df=df,
        cls_name=cls_name,
        acc_name=acc_name,
        dataset_name=dataset_name,
        basedir=basedir,
    )
