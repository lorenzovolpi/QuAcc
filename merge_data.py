import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from quacc.evaluation.estimators import CE
from quacc.evaluation.report import DatasetReport, DatasetReportInfo


def load_report_info(path: Path) -> DatasetReportInfo:
    return DatasetReport.unpickle(path, report_info=True)


def list_reports(base_path: Path | str):
    if isinstance(base_path, str):
        base_path = Path(base_path)

    if base_path.name == "plot":
        return []

    reports = []
    for f in os.listdir(base_path):
        fp = base_path / f
        if fp.is_dir():
            reports.extend(list_reports(fp))
        elif fp.is_file():
            if fp.suffix == ".pickle" and fp.stem == base_path.name:
                reports.append(load_report_info(fp))

    return reports


def playground():
    data_a = np.array(np.random.random((4, 6)))
    data_b = np.array(np.random.random((4, 4)))
    _ind1 = pd.MultiIndex.from_product([["0.2", "0.8"], ["0", "1"]])
    _col1 = pd.MultiIndex.from_product([["a", "b"], ["1", "2", "5"]])
    _col2 = pd.MultiIndex.from_product([["a", "b"], ["1", "2"]])
    a = pd.DataFrame(data_a, index=_ind1, columns=_col1)
    b = pd.DataFrame(data_b, index=_ind1, columns=_col2)
    print(a)
    print(b)
    print((a.index == b.index).all())
    update_col = a.columns.intersection(b.columns)
    col_to_join = b.columns.difference(update_col)
    _b = b.drop(columns=[(slice(None), "2")])
    _join = pd.concat([a, _b.loc[:, col_to_join]], axis=1)
    _join.loc[:, update_col.to_list()] = _b.loc[:, update_col.to_list()]
    _join.sort_index(axis=1, level=0, sort_remaining=False, inplace=True)

    print(_join)


def merge(dri1: DatasetReportInfo, dri2: DatasetReportInfo, path: Path):
    drm = dri1.dr.join(dri2.dr, estimators=CE.name.all)

    # save merged dr
    _path = path / drm.name / f"{drm.name}.pickle"
    os.makedirs(_path.parent, exist_ok=True)
    drm.pickle(_path)

    # rename dri1 pickle
    dri1_bp = Path(dri1.name) / f"{dri1.name.split('/')[-1]}.pickle"
    os.rename(dri1_bp, dri1_bp.with_suffix(f".pickle.pre_{dri2.name.split('/')[-2]}"))

    # copy merged pickle in place of old dri1 one
    shutil.copyfile(_path, dri1_bp)

    # copy dri2 log file inside dri1 folder
    dri2_bp = Path(dri2.name) / f"{dri2.name.split('/')[-1]}.pickle"
    shutil.copyfile(
        dri2_bp.with_suffix(".log"),
        dri1_bp.with_name(f"{dri1_bp.stem}_{dri2.name.split('/')[-2]}.log"),
    )


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("path1", nargs="?", default=None)
    parser.add_argument("path2", nargs="?", default=None)
    parser.add_argument("-l", "--list", action="store_true", dest="list")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose")
    parser.add_argument(
        "-o", "--output", action="store", dest="output", default="output/merge"
    )
    args = parser.parse_args()

    reports = list_reports("output")
    reports = {r.name: r for r in reports}

    if args.list:
        for i, r in enumerate(reports.values()):
            if args.verbose:
                print(f"{i}: {r}")
            else:
                print(f"{i}: {r.name}")
    else:
        dri1, dri2 = reports.get(args.path1, None), reports.get(args.path2, None)
        if dri1 is None or dri2 is None:
            raise ValueError(
                f"({args.path1}, {args.path2}) is not a valid pair of paths"
            )
        merge(dri1, dri2, path=Path(args.output))


if __name__ == "__main__":
    run()
