import itertools as IT
import os

import pandas as pd

from exp.leap.config import (
    PROBLEM,
    get_acc_names,
    get_baseline_names,
    get_classifier_names,
    get_dataset_names,
    get_method_names,
    root_dir,
)
from exp.leap.util import load_results
from quacc.table import Format, Table


def rename_dataset(df, datasets):
    mapping = {
        "poker_hand": "poker-hand",
        "hand_digits": "hand-digits",
        "page_block": "page-block",
        "image_seg": "image-seg",
    }

    _datasets = [mapping.get(d, d) for d in datasets]
    for d, rd in mapping.items():
        df.loc[df["dataset"] == d, "dataset"] = rd
    return df, _datasets


def rename_methods(df, methods, baselines):
    mapping = {
        "Naive": 'Na\\"ive',
        "PHD(KDEy)": "S-LEAP(KDEy)",
        "OCE(KDEy)-SLSQP": "O-LEAP(KDEy)",
    }

    _methods = [mapping.get(m, m) for m in methods]
    _baselines = [mapping.get(b, b) for b in baselines]
    for m, rm in mapping.items():
        df.loc[df["method"] == m, "method"] = rm
    return df, _methods, _baselines


def tables():
    res = load_results()

    def gen_table(df: pd.DataFrame, name, datasets, methods, baselines):
        tbl = Table(name=name, benchmarks=datasets, methods=methods, baselines=baselines)
        tbl.format = Format(
            mean_prec=4,
            show_std=True,
            remove_zero=True,
            with_rank_mean=False,
            with_mean=True,
            color=True,
            color_mode="baselines",
        )
        tbl.format.mean_macro = False
        for dataset, method in IT.product(datasets, methods):
            values = df.loc[(df["dataset"] == dataset) & (df["method"] == method), "acc_err"].to_numpy()
            for v in values:
                tbl.add(dataset, method, v)
        return tbl

    classifiers = get_classifier_names()
    datasets = get_dataset_names()
    methods = get_method_names()
    baselines = get_baseline_names()
    accs = get_acc_names()

    tbls = []
    for classifier, acc in IT.product(classifiers, accs):
        _df = res.loc[(res["classifier"] == classifier) & (res["acc_name"] == acc), :]
        name = f"{PROBLEM}_{classifier}_{acc}"
        _df, _datasets = rename_dataset(_df, datasets)
        _df, _methods, _baselines = rename_methods(_df, methods, baselines)
        tbls.append(gen_table(_df, name, _datasets, _methods, _baselines))

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
