import itertools as IT
from collections import defaultdict

import numpy as np
import pandas as pd

import quacc as qc
from quacc.experiments.generators import gen_acc_measure, gen_bin_datasets, gen_classifiers
from quacc.experiments.report import Report
from quacc.utils.table import Format, Table

PROBLEM = "binary"
ERROR = qc.error.se

if PROBLEM == "binary":
    gen_datasets = gen_bin_datasets


def table_from_df(df: pd.DataFrame, name, benchmarks, methods) -> Table:
    tbl = Table(name=name, benchmarks=benchmarks, methods=methods)
    tbl.format = Format(mean_prec=5, show_std=False, remove_zero=True)
    for dataset, method in IT.product(benchmarks, methods):
        values = df.loc[(df["dataset"] == dataset) & (df["method"] == method), ["acc_err"]].to_numpy()
        for v in values:
            tbl.add(dataset, method, v)

    return tbl


def get_method_cls_name(method, cls):
    return f"{method}-{cls}"


def gen_n2e_tables():
    pdf_path = f"tables/n2e/{PROBLEM}.pdf"

    def rename_method(m):
        methods_dict = {
            "N2E(ACC-h0)": "PhD(ACC)",
            "N2E(KDEy-h0)": "PhD(KDEy)",
        }
        return methods_dict.get(m, m)

    def rename_cls(cls):
        cls_dict = {"SVM(rbf)": "SVM"}
        return cls_dict.get(cls, cls)

    benchmarks = [name for name, _ in gen_datasets(only_names=True)]
    base_methods = ["Naive", "ATC-MC", "DoC", "N2E(ACC-h0)", "N2E(KDEy-h0)"]
    tbl_methods = [rename_method(m) for m in base_methods]
    base_cls = [cls_name for cls_name, _ in gen_classifiers()]
    acc_names = [acc_name for acc_name, _ in gen_acc_measure()]

    tables = []
    for acc_name in acc_names:
        for cls_name in base_cls:
            rep = Report.load_results(PROBLEM, cls_name, acc_name, benchmarks, base_methods)
            df = rep.table_data(mean=False, error=ERROR)

            # rename methods and cls
            cls_name = rename_cls(cls_name)
            for m in base_methods:
                df.loc[df["method"] == m, ["method"]] = rename_method(m)

            # build table
            tbl_name = f"{PROBLEM}_{cls_name}_{acc_name}"
            tbl = table_from_df(df, name=tbl_name, benchmarks=benchmarks, methods=tbl_methods)
            tables.append(tbl)

    # Table.LatexPDF(pdf_path=pdf_path, tables=tables, landscape=False)
    for t in tables:
        print(t.tabular())
        print()


if __name__ == "__main__":
    gen_n2e_tables()
