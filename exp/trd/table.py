import itertools as IT
import os

import pandas as pd

from exp.trd.config import PROBLEM, get_acc_names, root_dir
from exp.trd.util import load_results, model_selection
from quacc.table import Format, Table


def tables():
    res = load_results()
    res = model_selection(res, oracle=False, only_default=True)

    accs = get_acc_names()
    methods = res["method"].unique().tolist()
    datasets = res["dataset"].unique().tolist()

    def gen_table(df: pd.DataFrame, name, datasets, methods):
        tbl = Table(name=name, benchmarks=datasets, methods=methods)
        tbl.format = Format(
            lower_is_better=False,
            mean_prec=3,
            show_std=True,
            remove_zero=True,
            with_rank_mean=False,
            with_mean=True,
            mean_macro=False,
            color=True,
            color_mode="local",
            simple_stat=True,
        )
        for dataset, method in IT.product(datasets, methods):
            values = df.loc[(df["dataset"] == dataset) & (df["method"] == method), "true_accs"].to_numpy()
            for v in values:
                tbl.add(dataset, method, v)
        return tbl

    tbls = []
    for acc in accs:
        _df = res.loc[res["acc_name"] == acc, :]
        name = f"{PROBLEM}_{acc}"
        tbls.append(gen_table(_df, name, datasets, methods))

    pdf_path = os.path.join(root_dir, "tables", f"{PROBLEM}.pdf")
    Table.LatexPDF(pdf_path, tables=tbls, landscape=False)


if __name__ == "__main__":
    tables()
