import itertools as IT
import os
from collections import defaultdict
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

import quacc as qc
from quacc.experiments.generators import gen_acc_measure, gen_bin_datasets, gen_classifiers
from quacc.experiments.report import Report
from quacc.utils.table import Format, Table

PROBLEM = "binary"
ERROR = qc.error.ae

if PROBLEM == "binary":
    gen_datasets = gen_bin_datasets


def table_from_df(df: pd.DataFrame, name, benchmarks, methods) -> Table:
    tbl = Table(name=name, benchmarks=benchmarks, methods=methods)
    tbl.format = Format(mean_prec=4, show_std=False, remove_zero=True, with_rank_mean=False)
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
            "Naive": "\\naive",
            "N2E(ACC-h0)": "\\phd",
            "N2E(KDEy-h0)": "\\phdplus",
        }
        return methods_dict.get(m, m)

    def rename_cls(cls):
        cls_dict = {"SVM(rbf)": "SVM", "KNN_10": "$k$-NN"}
        return cls_dict.get(cls, cls)

    benchmarks = [name for name, _ in gen_datasets(only_names=True)]
    base_methods = ["Naive", "ATC-MC", "DoC", "N2E(ACC-h0)", "N2E(KDEy-h0)"]
    tbl_methods = [rename_method(m) for m in base_methods]
    # base_cls = [cls_name for cls_name, _ in gen_classifiers()]
    base_cls = ["LR", "KNN_10", "SVM(rbf)"]
    acc_names = [acc_name for acc_name, _ in gen_acc_measure()]

    tables = []
    for acc_name in acc_names:
        for cls_name in base_cls:
            rep = Report.load_results(PROBLEM, cls_name, acc_name, benchmarks, base_methods)
            df = rep.table_data(mean=False, error=ERROR)

            # rename methods and cls
            cls_name = rename_cls(cls_name)
            # for m in base_methods:
            #     df.loc[df["method"] == m, ["method"]] = rename_method(m)

            # build table
            tbl_name = f"{PROBLEM}_{cls_name}_{acc_name}"
            tbl = table_from_df(df, name=tbl_name, benchmarks=benchmarks, methods=base_methods)
            tables.append(tbl)

    Table.LatexPDF(pdf_path=pdf_path, tables=tables, landscape=False)

    tbl_dict = defaultdict(lambda: [])
    tbl_endline = defaultdict(lambda: "\\\\")
    for t in tables:
        tabular = StringIO(t.tabular())
        for ln in tabular:
            ln = ln.strip()

            if ln.startswith("\\begin{tabular}"):
                continue
            if ln.startswith("\\end{tabular}"):
                continue
            if ln.startswith("\\cline"):
                continue
            if ln.startswith("\\multicolumn"):
                continue

            amp_idx = ln.find("&")
            name = ln[:amp_idx].strip()
            row = ln[amp_idx + 1 :]
            if row.endswith("\\\\"):
                row = row[:-2].strip()
            elif row.endswith("\\\\\\hline"):
                row = row[:-8].strip()
                tbl_endline[name] = "\\\\\\hline"
            tbl_dict[name].append(row)

    corpus = []
    for name, row_l in tbl_dict.items():
        row = " & ".join(row_l)
        corpus.append(f"{name} & {row} {tbl_endline[name]}")
    corpus = "\n".join(corpus) + "\n"
    begin = "\\begin{tabular}{|c|" + (("c" * len(base_methods)) + "|") * len(base_cls) + "}\n"
    end = "\\end{tabular}\n"
    cline = "\\cline{2-" + str(len(base_methods) * len(base_cls) + 1) + "}\n"
    multicol1 = (
        "\\multicolumn{1}{c|}{} & "
        + " & ".join(["\\multicolumn{" + str(len(base_methods)) + "}{c|}{" + rename_cls(cls) + "}" for cls in base_cls])
        + " \\\\\n"
    )
    multicol2 = (
        "\\multicolumn{1}{c|}{} & "
        + " & ".join([" & ".join([m for m in tbl_methods]) for _ in base_cls])
        + " \\\\\\hline\n"
    )

    hstack_path = os.path.join(str(Path(pdf_path).parent), f"{PROBLEM}_hstack.tex")
    with open(hstack_path, "w") as f:
        f.write(begin + cline + multicol1 + cline + multicol2 + corpus + end)


if __name__ == "__main__":
    gen_n2e_tables()
