import itertools as IT
import os

import pandas as pd

from exp.leap.config import PROBLEM, root_dir
from exp.leap.generators import get_dataset_names, get_method_names
from exp.leap.util import load_results
from quacc.table import Format, Table


def tables():
    res = load_results()

    def gen_table(df: pd.DataFrame, name, datasets, methods):
        tbl = Table(name=name, benchmarks=datasets, methods=methods)
        tbl.format = Format(
            mean_prec=4, show_std=True, remove_zero=True, with_rank_mean=False, with_mean=True, color=True
        )
        tbl.format.mean_macro = False
        for dataset, method in IT.product(datasets, methods):
            values = df.loc[(df["dataset"] == dataset) & (df["method"] == method), "acc_err"].to_numpy()
            for v in values:
                tbl.add(dataset, method, v)
        return tbl

    classifiers = res["classifier"].unique()
    datasets = get_dataset_names()
    methods = get_method_names()
    accs = res["acc_name"].unique()

    tbls = []
    for classifier, acc in IT.product(classifiers, accs):
        _df = res.loc[(res["classifier"] == classifier) & (res["acc_name"] == acc), :]
        name = f"{PROBLEM}_{classifier}_{acc}"
        tbls.append(gen_table(_df, name, datasets, methods))

    pdf_path = os.path.join(root_dir, "tables", f"{PROBLEM}.pdf")
    Table.LatexPDF(pdf_path, tables=tbls, landscape=False)


def leap_true_solve():
    res = load_results()
    methods = ["LEAP(KDEy)", "LEAP(KDEy-a)", "LEAP(MDy)"]
    md_path = os.path.join(root_dir, "tables", f"{PROBLEM}_true_solve.md")

    pd.pivot_table(
        res.loc[res["method"].isin(methods)],
        columns=["classifier", "method"],
        index=["dataset"],
        values="true_solve",
    ).to_markdown(md_path)


if __name__ == "__main__":
    tables()
    # leap_true_solve()
