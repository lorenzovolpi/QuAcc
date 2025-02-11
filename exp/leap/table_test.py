import itertools as IT
import os

import pandas as pd

from exp.leap.config import root_dir
from quacc.table import Format, Table

data = [
    [1, 2, 1, 2, 0],
    [6, 4, 0, 1, 2],
    [4, 3, 1, 2, 0],
]


datasets = ["x", "y", "z"]
methods = ["a", "b", "c", "d", "e"]
baselines = ["a", "b"]
out_methods = [m for m in methods if m not in baselines]

df = pd.DataFrame(data, columns=methods, index=datasets)

if __name__ == "__main__":
    name = "test"
    tbl = Table(name=name, benchmarks=datasets, methods=methods)
    tbl.format = Format(mean_prec=4, show_std=True, remove_zero=True, with_rank_mean=False, with_mean=True, color=True)
    tbl.format.mean_macro = False
    for dataset, method in IT.product(datasets, methods):
        val = df.loc[dataset, method]
        tbl.add(dataset, method, val)

    pdf_path = os.path.join(root_dir, "tables", "test_table.pdf")
    Table.LatexPDF(pdf_path, tables=[tbl], landscape=False)
