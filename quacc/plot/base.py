from pathlib import Path


class BasePlot:
    @classmethod
    def save_fig(cls, fig, base_path, title) -> Path:
        ...

    @classmethod
    def plot_diagonal(
        cls,
        reference,
        columns,
        data,
        *,
        pos_class=1,
        title="default",
        x_label="true",
        y_label="estim.",
        legend=True,
    ):
        ...

    @classmethod
    def plot_delta(
        cls,
        base_prevs,
        columns,
        data,
        *,
        stdevs=None,
        pos_class=1,
        title="default",
        x_label="prevs.",
        y_label="error",
        legend=True,
    ):
        ...

    @classmethod
    def plot_shift(
        cls,
        shift_prevs,
        columns,
        data,
        *,
        counts=None,
        pos_class=1,
        title="default",
        x_label="true",
        y_label="estim.",
        legend=True,
    ):
        ...

    @classmethod
    def plot_fit_scores(
        train_prevs,
        scores,
        *,
        pos_class=1,
        title="default",
        x_label="prev.",
        y_label="position",
        legend=True,
    ):
        ...
