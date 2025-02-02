import itertools as IT
import os
from traceback import print_exception

import numpy as np
import pandas as pd

from cap.experiments.run import (
    CSV_SEP,
    PROBLEM,
    get_dataset_names,
    get_method_names,
    log,
    root_dir,
)
from cap.experiments.util import load_results
from cap.table import Format, Table


def tables(df: pd.DataFrame):
    def _sort(ls: np.ndarray | list, cat) -> list:
        ls = np.array(ls)

        if cat == "m":
            original_ls = np.array(get_method_names())
        elif cat == "b":
            original_ls = np.array(get_dataset_names())
        else:
            return ls.tolist()

        original_ls = np.append(original_ls, ls[~np.isin(ls, original_ls)])
        orig_idx = np.argsort(original_ls)
        sorted_idx = np.searchsorted(original_ls[orig_idx], ls)

        return original_ls[np.sort(orig_idx[sorted_idx])].tolist()

    def gen_table(df: pd.DataFrame, name, benchmarks, methods, acc_names):
        acc_name_map = {"vanilla_accuracy": "acc", "macro-F1": "f1"}
        bench_acc_map = {
            (b, a): f"{b}-{acc_name_map[a]}"
            for b, a in IT.product(benchmarks, acc_names)
        }

        tbl = Table(name=name, benchmarks=list(bench_acc_map.values()), methods=methods)
        tbl.format = Format(
            mean_prec=4,
            show_std=True,
            remove_zero=True,
            with_rank_mean=False,
            with_mean=False,
            color=False,
            show_stat=False,
            stat_test="wilcoxon",
        )
        tbl.format.mean_macro = False
        for (dataset, acc), method in IT.product(list(bench_acc_map.keys()), methods):
            values = df.loc[
                (df["dataset"] == dataset)
                & (df["method"] == method)
                & (df["acc_name"] == acc),
                ["acc_err"],
            ].to_numpy()
            for v in values:
                tbl.add(bench_acc_map[(dataset, acc)], method, v)
        return tbl

    pdf_path = os.path.join(root_dir, "tables", f"{PROBLEM}.pdf")

    acc_names = [
        "vanilla_accuracy",
        "macro-F1",
    ]
    configs = [
        {
            "name": "all",
            "benchmarks": df["dataset"].unique(),
            "methods": [m for m in df["method"].unique() if not m == "LEAP(SLD)"],
        },
        {
            "name": "paper",
            "benchmarks": df["dataset"].unique(),
            "methods": [
                m
                for m in df["method"].unique()
                if not (
                    m.endswith("1xn2")
                    or m.endswith("1xnp1")
                    or m.endswith("nxn")
                    or m == "LEAP(SLD)"
                )
            ],
        },
    ]
    configs = [
        d
        | {
            "methods": _sort(d["methods"], "m"),
            "benchmarks": _sort(d["benchmarks"], "b"),
        }
        for d in configs
    ]

    tables = []
    for config in configs:
        for cls_name in df["classifier"].unique():
            _df = df.loc[df["classifier"] == cls_name]
            # build table
            tbl_name = f"{PROBLEM}_{cls_name}_{config['name']}"
            tbl = gen_table(
                _df,
                name=tbl_name,
                benchmarks=config["benchmarks"],
                methods=config["methods"],
                acc_names=acc_names,
            )
            log.info(f"Table for config={config['name']} - cls={cls_name} generated")
            tables.append(tbl)

    Table.LatexPDF(pdf_path=pdf_path, tables=tables, landscape=False, transpose=True)
    log.info("Pdf table summary generated")


if __name__ == "__main__":
    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        results = load_results(PROBLEM, root_dir, CSV_SEP)
        tables(results)
        log.info("-" * 32 + "  end  " + "-" * 32)
    except Exception as e:
        log.error(e)
        print_exception(e)
