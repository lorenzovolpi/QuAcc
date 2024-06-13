import itertools as IT

import pandas as pd

import quacc as qc
from quacc.experiments.generators import gen_acc_measure, gen_bin_datasets, gen_classifiers
from quacc.experiments.report import Report
from quacc.utils.table import Table

PROBLEM = "binary"
ERROR = qc.error.se

if PROBLEM == "binary":
    gen_datasets = gen_bin_datasets


def table_from_df(df: pd.DataFrame, name, benchmarks, methods) -> Table:
    tbl = Table(name=name, benchmarks=benchmarks, methods=methods)
    for dataset, method in IT.product(benchmarks, methods):
        values = df.loc[(df["dataset"] == dataset) & (df["method"] == method), ["acc_err"]].to_numpy()
        for v in values:
            tbl.add(dataset, method, v)

    return tbl


def gen_n2e_tables():
    pdf_path = f"tables/n2e/{PROBLEM}.pdf"
    methods_dict = {
        "ATC-MC": None,
        "DoC": None,
        "Naive": None,
        "N2E(ACC-h0)": "PhD(ACC-h0)",
        "N2E(KDEy-h0)": "PhD(KDEy-h0)",
    }
    methods_dict = {k: k if v is None else v for k, v in methods_dict.items()}
    benchmarks = [name for name, _ in gen_datasets(only_names=True)]

    tables = []
    for acc_name, _ in gen_acc_measure():
        for cls_name, _ in gen_classifiers():
            # load dataframe
            rep = Report.load_results(PROBLEM, cls_name, acc_name, benchmarks, list(methods_dict.keys()))
            df = rep.table_data(mean=False, error=ERROR)

            # rename methods
            for mn_old, mn_new in methods_dict.items():
                df.loc[df["method"] == mn_old, ["method"]] = mn_new

            # build table
            tbl_name = f"{PROBLEM}_{cls_name}_{acc_name}"
            tbl = table_from_df(df, name=tbl_name, benchmarks=benchmarks, methods=list(methods_dict.values()))
            tables.append(tbl)

    Table.LatexPDF(pdf_path=pdf_path, tables=tables)


if __name__ == "__main__":
    gen_n2e_tables()
