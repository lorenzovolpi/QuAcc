import itertools as IT
import os

import seaborn as sns

from exp.leap.config import PROBLEM, get_acc_names, get_classifier_names, get_dataset_names, root_dir
from exp.leap.util import load_results, rename_datasets, rename_methods
from quacc.plot.seaborn import plot_diagonal_grid

method_map = {
    "LEAP(KDEy)": "LEAP$_{\\mathrm{KDEy}}$",
    "PHD(KDEy)": "S-LEAP$_{\\mathrm{KDEy}}$",
    "OCE(KDEy)-SLSQP": "O-LEAP$_{\\mathrm{KDEy}}$",
}

dataset_map = {
    "poker_hand": "poker-hand",
    "hand_digits": "hand-digits",
    "page_block": "page-block",
    "image_seg": "image-seg",
}


def get_selection_datasets():
    if PROBLEM == "binary":
        return ["haberman", "pageblocks.5", "iris.2", "yeast"]
    elif PROBLEM == "multiclass":
        return ["phishing", "page_block", "academic-success", "mhr"]


def plots():
    res = load_results()

    classifiers = get_classifier_names()
    accs = get_acc_names()
    methods = ["DoC", "LEAP(KDEy)", "PHD(KDEy)", "OCE(KDEy)-SLSQP"]

    dataset_configs = {
        "all": get_dataset_names(),
        "4x1": get_selection_datasets(),
    }

    parent_dir = os.path.join(root_dir, "plots")
    os.makedirs(parent_dir, exist_ok=True)

    for cls_name, acc in IT.product(classifiers[:1], accs):
        print(f"Plotting {cls_name} for {acc}")

        df = res.loc[(res["classifier"] == cls_name) & (res["acc_name"] == acc) & (res["method"].isin(methods)), :]
        df, _methods = rename_methods(method_map, df, methods)

        for config, datasets in dataset_configs.items():
            df, _datasets = rename_datasets(dataset_map, df, datasets)
            df_config = df.loc[df["dataset"].isin(_datasets), :]
            plot_diagonal_grid(
                df_config,
                _methods,
                basedir=parent_dir,
                filename=f"grid_{cls_name}_{PROBLEM}_{config}",
                n_cols=4,
                legend_bbox_to_anchor=(0.95, 0.3),
                palette="deep",
                # palette=sns.color_palette("hls", len(_methods)),
                x_label="True Accuracy",
                y_label="Estimated Accuracy",
            )


if __name__ == "__main__":
    plots()
