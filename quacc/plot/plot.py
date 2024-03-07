from quacc.plot.base import BasePlot
from quacc.plot.mpl import MplPlot
from quacc.plot.plotly import PlotlyPlot

__backend: BasePlot = MplPlot()


def get_backend(name, theme=None):
    match name:
        case "matplotlib" | "mpl":
            return MplPlot()
        case "plotly":
            return PlotlyPlot(theme=theme)
        case _:
            return MplPlot()


def plot_delta(
    base_prevs,
    columns,
    data,
    *,
    stdevs=None,
    pos_class=1,
    metric="acc",
    name="default",
    train_prev=None,
    legend=True,
    avg=None,
    save_fig=False,
    base_path=None,
    backend=None,
):
    backend = __backend if backend is None else backend
    _base_title = "delta_stdev" if stdevs is not None else "delta"
    if train_prev is not None:
        t_prev_pos = int(round(train_prev[pos_class] * 100))
        title = f"{_base_title}_{name}_{t_prev_pos}_{metric}"
    else:
        title = f"{_base_title}_{name}_avg_{avg}_{metric}"

    if avg is None or avg == "train":
        x_label = "Test Prevalence"
    else:
        x_label = "Train Prevalence"
    if metric == "acc":
        y_label = "Prediction Error for Vanilla Accuracy"
    elif metric == "f1":
        y_label = "Prediction Error for F1"
    else:
        y_label = f"{metric} error"
    fig = backend.plot_delta(
        base_prevs,
        columns,
        data,
        stdevs=stdevs,
        pos_class=pos_class,
        title=title,
        x_label=x_label,
        y_label=y_label,
        legend=legend,
    )

    if save_fig:
        output_path = backend.save_fig(fig, base_path, title)
        return fig, output_path

    return fig


def plot_diagonal(
    reference,
    columns,
    data,
    *,
    pos_class=1,
    metric="acc",
    name="default",
    train_prev=None,
    fixed_lim=False,
    legend=True,
    save_fig=False,
    base_path=None,
    backend=None,
):
    backend = __backend if backend is None else backend
    if train_prev is not None:
        t_prev_pos = int(round(train_prev[pos_class] * 100))
        title = f"diagonal_{name}_{t_prev_pos}_{metric}"
    else:
        title = f"diagonal_{name}_{metric}"

    if metric == "acc":
        x_label = "True Vanilla Accuracy"
        y_label = "Estimated Vanilla Accuracy"
    else:
        x_label = f"true {metric}"
        y_label = f"estim. {metric}"
    fig = backend.plot_diagonal(
        reference,
        columns,
        data,
        pos_class=pos_class,
        title=title,
        x_label=x_label,
        y_label=y_label,
        fixed_lim=fixed_lim,
        legend=legend,
    )

    if save_fig:
        output_path = backend.save_fig(fig, base_path, title)
        return fig, output_path

    return fig


def plot_shift(
    shift_prevs,
    columns,
    data,
    *,
    counts=None,
    pos_class=1,
    metric="acc",
    name="default",
    train_prev=None,
    legend=True,
    save_fig=False,
    base_path=None,
    backend=None,
):
    backend = __backend if backend is None else backend
    if train_prev is not None:
        t_prev_pos = int(round(train_prev[pos_class] * 100))
        title = f"shift_{name}_{t_prev_pos}_{metric}"
    else:
        title = f"shift_{name}_avg_{metric}"

    x_label = "Amount of Prior Probability Shift"
    if metric == "acc":
        y_label = "Prediction Error for Vanilla Accuracy"
    elif metric == "f1":
        y_label = "Prediction Error for F1"
    else:
        y_label = f"{metric} error"
    fig = backend.plot_shift(
        shift_prevs,
        columns,
        data,
        counts=counts,
        pos_class=pos_class,
        title=title,
        x_label=x_label,
        y_label=y_label,
        legend=legend,
    )

    if save_fig:
        output_path = backend.save_fig(fig, base_path, title)
        return fig, output_path

    return fig


def plot_fit_scores(
    train_prevs,
    scores,
    *,
    pos_class=1,
    metric="acc",
    name="default",
    legend=True,
    save_fig=False,
    base_path=None,
    backend=None,
):
    backend = __backend if backend is None else backend
    title = f"fit_scores_{name}_avg_{metric}"

    x_label = "train prev."
    y_label = "position"
    fig = backend.plot_fit_scores(
        train_prevs,
        scores,
        pos_class=pos_class,
        title=title,
        x_label=x_label,
        y_label=y_label,
        legend=legend,
    )

    if save_fig:
        output_path = backend.save_fig(fig, base_path, title)
        return fig, output_path

    return fig
