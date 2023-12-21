from pathlib import Path
from re import X

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from sklearn import base

from quacc import utils
from quacc.plot.base import BasePlot

matplotlib.use("agg")


class MplPlot(BasePlot):
    def _get_markers(self, n: int):
        ls = "ovx+sDph*^1234X><.Pd"
        if n > len(ls):
            ls = ls * (n / len(ls) + 1)
        return list(ls)[:n]

    def save_fig(self, fig, base_path, title) -> Path:
        if base_path is None:
            base_path = utils.get_quacc_home() / "plots"
        output_path = base_path / f"{title}.png"
        fig.savefig(output_path, bbox_inches="tight")
        return output_path

    def plot_delta(
        self,
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
        fig, ax = plt.subplots()
        ax.set_aspect("auto")
        ax.grid()

        NUM_COLORS = len(data)
        cm = plt.get_cmap("tab10")
        if NUM_COLORS > 10:
            cm = plt.get_cmap("tab20")
        cy = cycler(color=[cm(i) for i in range(NUM_COLORS)])

        # base_prevs = base_prevs[:, pos_class]
        if isinstance(base_prevs[0], float):
            base_prevs = np.around([(1 - bp, bp) for bp in base_prevs], decimals=4)
        str_base_prevs = [str(tuple(bp)) for bp in base_prevs]
        # xticks = [str(bp) for bp in base_prevs]
        xticks = np.arange(len(base_prevs))
        for method, deltas, _cy in zip(columns, data, cy):
            ax.plot(
                xticks,
                deltas,
                label=method,
                color=_cy["color"],
                linestyle="-",
                marker="o",
                markersize=3,
                zorder=2,
            )
            if stdevs is not None:
                _col_idx = np.where(columns == method)[0]
                stdev = stdevs[_col_idx].flatten()
                nn_idx = np.intersect1d(
                    np.where(deltas != np.nan)[0],
                    np.where(stdev != np.nan)[0],
                )
                _bps, _ds, _st = xticks[nn_idx], deltas[nn_idx], stdev[nn_idx]
                ax.fill_between(
                    _bps,
                    _ds - _st,
                    _ds + _st,
                    color=_cy["color"],
                    alpha=0.25,
                )

        def format_fn(tick_val, tick_pos):
            if int(tick_val) in xticks:
                return str_base_prevs[int(tick_val)]

            return ""

        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6, integer=True, prune="both"))
        ax.xaxis.set_major_formatter(format_fn)

        ax.set(
            xlabel=f"{x_label} prevalence",
            ylabel=y_label,
            title=title,
        )

        if legend:
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        return fig

    def plot_diagonal(
        self,
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
        fig, ax = plt.subplots()
        ax.set_aspect("auto")
        ax.grid()
        ax.set_aspect("equal")

        NUM_COLORS = len(data)
        cm = plt.get_cmap("tab10")
        if NUM_COLORS > 10:
            cm = plt.get_cmap("tab20")
        cy = cycler(
            color=[cm(i) for i in range(NUM_COLORS)],
            marker=self._get_markers(NUM_COLORS),
        )

        reference = np.array(reference)
        x_ticks = np.unique(reference)
        x_ticks.sort()

        for deltas, _cy in zip(data, cy):
            ax.plot(
                reference,
                deltas,
                color=_cy["color"],
                linestyle="None",
                marker=_cy["marker"],
                markersize=3,
                zorder=2,
                alpha=0.25,
            )

        # ensure limits are equal for both axes
        _alims = np.stack(((ax.get_xlim(), ax.get_ylim())), axis=-1)
        _lims = np.array([f(ls) for f, ls in zip([np.min, np.max], _alims)])
        ax.set(xlim=tuple(_lims), ylim=tuple(_lims))

        for method, deltas, _cy in zip(columns, data, cy):
            slope, interc = np.polyfit(reference, deltas, 1)
            y_lr = np.array([slope * x + interc for x in _lims])
            ax.plot(
                _lims,
                y_lr,
                label=method,
                color=_cy["color"],
                linestyle="-",
                markersize="0",
                zorder=1,
            )

        # plot reference line
        ax.plot(
            _lims,
            _lims,
            color="black",
            linestyle="--",
            markersize=0,
            zorder=1,
        )

        ax.set(xlabel=x_label, ylabel=y_label, title=title)

        if legend:
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        return fig

    def plot_shift(
        self,
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
        fig, ax = plt.subplots()
        ax.set_aspect("auto")
        ax.grid()

        NUM_COLORS = len(data)
        cm = plt.get_cmap("tab10")
        if NUM_COLORS > 10:
            cm = plt.get_cmap("tab20")
        cy = cycler(color=[cm(i) for i in range(NUM_COLORS)])

        # shift_prevs = shift_prevs[:, pos_class]
        for method, shifts, _cy in zip(columns, data, cy):
            ax.plot(
                shift_prevs,
                shifts,
                label=method,
                color=_cy["color"],
                linestyle="-",
                marker="o",
                markersize=3,
                zorder=2,
            )
            if counts is not None:
                _col_idx = np.where(columns == method)[0]
                count = counts[_col_idx].flatten()
                for prev, shift, cnt in zip(shift_prevs, shifts, count):
                    label = f"{cnt}"
                    plt.annotate(
                        label,
                        (prev, shift),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        color=_cy["color"],
                        fontsize=12.0,
                    )

        ax.set(xlabel=x_label, ylabel=y_label, title=title)

        if legend:
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        return fig
