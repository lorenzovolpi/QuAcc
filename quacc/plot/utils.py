import numpy as np
import plotly.graph_objects as go

from quacc.utils.commons import get_plots_path


def _get_ref_limits(true_accs: np.ndarray, estim_accs: np.ndarray):
    """get lmits of reference line"""

    _edges = (
        np.min([np.min(true_accs), np.min(estim_accs)]),
        np.max([np.max(true_accs), np.max(estim_accs)]),
    )
    _lims = np.array([[_edges[0], _edges[1]], [_edges[0], _edges[1]]])
    return _lims
