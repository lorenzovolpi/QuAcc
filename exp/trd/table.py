import itertools as IT
import os

import pandas as pd

from exp.trd.config import PROBLEM, get_acc_names, root_dir
from exp.trd.model_selection import model_selection
from exp.trd.util import decorate_datasets, load_results, rename_datasets, rename_methods
from quacc.table import Format, Table

method_map = {
    "Naive": 'Na\\"ive',
    "ATC-MC": "ATC",
    "LEAP(KDEy)": "\\leapplus",
    "S-LEAP(KDEy)": "\\leapppskde",
    "O-LEAPCE(KDEy)-SLSQP": "\\oleapkde",
}

dataset_map = {
    "poker_hand": "poker-hand",
    "hand_digits": "hand-digits",
    "page_block": "page-block",
    "image_seg": "image-seg",
}


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
        _df, _datasets = rename_datasets(dataset_map, _df, datasets)
        _df, _datasets = decorate_datasets(_df, _datasets)
        _df, _methods = rename_methods(method_map, _df, methods)
        tbls.append(gen_table(_df, name, _datasets, _methods))

    pdf_path = os.path.join(root_dir, "tables", f"{PROBLEM}.pdf")
    Table.LatexPDF(pdf_path, tables=tbls, landscape=False)


if __name__ == "__main__":
    tables()
