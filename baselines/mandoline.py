from functools import partial
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import scipy.optimize
import scipy.special
import sklearn.metrics.pairwise as skmetrics


def Phi(
    D: np.ndarray,
    edge_list: np.ndarray = None,
):
    """
    Given an n x d matrix of (example, slices), calculate the potential
    matrix.

    Includes correlations modeled by the edges in the `edge_list`.

    Args:
        D (np.ndarray): n x d matrix of (example, slice)
        edge_list (np.ndarray): k x 2 matrix of edge correlations to be modeled.
            edge_list[i, :] should be indices for a pair of columns of D.

    Returns:
        Potential matrix. Equals D when edge_list is None, otherwise adds additional
        (x_i * x_j) "cross-terms" corresponding to the edges in the `edge_list`.

    Examples:
        >>> D = np.random.choice([-1, 1], size=(100, 6))
        >>> edge_list = np.array([(0, 1), (1, 4)])
        >>> Phi(D, edge_list)
    """

    if edge_list is not None:
        pairwise_terms = (
            D[np.arange(len(D)), edge_list[:, 0][:, np.newaxis]].T
            * D[np.arange(len(D)), edge_list[:, 1][:, np.newaxis]].T
        )
        return np.concatenate([D, pairwise_terms], axis=1)
    else:
        return D


def log_partition_ratio(
    x: np.ndarray,
    Phi_D_src: np.ndarray,
    n_src: int,
):
    """
    Calculate the log-partition ratio in the KLIEP problem.
    """
    return np.log(n_src) - scipy.special.logsumexp(Phi_D_src.dot(x))


def mandoline(
    D_src: np.ndarray,
    D_tgt: np.ndarray,
    edge_list: np.ndarray,
    sigma: float = None,
):
    """
    Mandoline solver.

    Args:
        D_src: (n_src x d) matrix of (example, slices) for the source distribution.
        D_tgt: (n_tgt x d) matrix of (example, slices) for the source distribution.
        edge_list: list of edge correlations between slices that should be modeled.
        sigma: optional parameter that activates RBF kernel-based KLIEP with scale
        `sigma`.

    Returns: SimpleNamespace that contains
        opt: result of scipy.optimize
        Phi_D_src: source potential matrix used in Mandoline
        Phi_D_tgt: target potential matrix used in Mandoline
        n_src: number of source samples
        n_tgt: number of target samples
        edge_list: the `edge_list` parameter passed as input

    """
    # Copy and binarize the input matrices to -1/1
    D_src, D_tgt = np.copy(D_src), np.copy(D_tgt)
    if np.min(D_src) == 0:
        D_src[D_src == 0] = -1
        D_tgt[D_tgt == 0] = -1

    # Edge list encoding dependencies between gs
    if edge_list is not None:
        edge_list = np.array(edge_list)

    # Create the potential matrices
    Phi_D_tgt, Phi_D_src = Phi(D_tgt, edge_list), Phi(D_src, edge_list)

    # Number of examples
    n_src, n_tgt = Phi_D_src.shape[0], Phi_D_tgt.shape[0]

    def f(x):
        obj = Phi_D_tgt.dot(x).sum() - n_tgt * scipy.special.logsumexp(Phi_D_src.dot(x))
        return -obj

    # Set the kernel
    kernel = partial(skmetrics.rbf_kernel, gamma=sigma)

    def llkliep_f(x):
        obj = kernel(
            Phi_D_tgt, x[:, np.newaxis]
        ).sum() - n_tgt * scipy.special.logsumexp(kernel(Phi_D_src, x[:, np.newaxis]))
        return -obj

    # Solve
    if not sigma:
        opt = scipy.optimize.minimize(
            f, np.random.randn(Phi_D_tgt.shape[1]), method="BFGS"
        )
    else:
        opt = scipy.optimize.minimize(
            llkliep_f, np.random.randn(Phi_D_tgt.shape[1]), method="BFGS"
        )

    return SimpleNamespace(
        opt=opt,
        Phi_D_src=Phi_D_src,
        Phi_D_tgt=Phi_D_tgt,
        n_src=n_src,
        n_tgt=n_tgt,
        edge_list=edge_list,
    )


def log_density_ratio(D, solved):
    """
    Calculate the log density ratio for a solved Mandoline run.
    """
    Phi_D = Phi(D, None)
    return Phi_D.dot(solved.opt.x) + log_partition_ratio(
        solved.opt.x, solved.Phi_D_src, solved.n_src
    )


def get_k_most_unbalanced_gs(D_src, D_tgt, k):
    """
    Get the top k slices that shift most between source and target
    distributions.

    Uses difference in marginals between each slice.
    """
    marginal_diff = np.abs(D_src.mean(axis=0) - D_tgt.mean(axis=0))
    differences = np.sort(marginal_diff)[-k:]
    indices = np.argsort(marginal_diff)[-k:]
    return list(indices), list(differences)


def weighted_estimator(weights: Optional[np.ndarray], mat: np.ndarray):
    """
    Calculate a weighted empirical mean over a matrix of samples.

    Args:
        weights (Optional[np.ndarray]):
            length n array of weights that sums to 1. Calculates an unweighted
            mean if `weights` is None.
        mat (np.ndarray):
            (n x r) matrix of empirical observations that is being averaged.

    Returns:
        Length r np.ndarray of weighted means.
    """
    _sum_weights = np.sum(weights)
    if _sum_weights != 1.0:
        if (_err := abs(1.0 - _sum_weights)) > 1e-15:
            print(_err)
            assert _sum_weights == 1, "`weights` must sum to 1."

    if weights is None:
        return np.mean(mat, axis=0)
    return np.sum(weights[:, np.newaxis] * mat, axis=0)


def estimate_performance(
    D_src: np.ndarray,
    D_tgt: np.ndarray,
    edge_list: np.ndarray,
    empirical_mat_list_src: List[np.ndarray],
):
    """
    Estimate performance on a target distribution using slice information from the
    source and target data.

    This function runs Mandoline to calculate the importance weights to reweight
    the source data.

    Args:
        D_src (np.ndarray): (n_src x d) matrix of (example, slices) for the source
            distribution.
        D_tgt (np.ndarray): (n_tgt x d) matrix of (example, slices) for the target
            distribution.
        edge_list (np.ndarray):
        empirical_mat_list_src (List[np.ndarray]):

    Returns:
        SimpleNamespace with 3 attributes
        - `all_estimates` is a list of SimpleNamespace objects with
            2 attributes
            - `weighted` is the estimate for the target distribution
            - `source` is the estimate for the source distribution
        - `solved`: result of scipy.optimize Mandoline solver
        - `weights`: self-normalized importance weights used to weight the source data
    """
    # Run the solver
    solved = mandoline(D_src, D_tgt, edge_list)

    # Compute the weights on the source dataset
    density_ratios = np.e ** log_density_ratio(solved.Phi_D_src, solved)

    # Self-normalized importance weights
    weights = density_ratios / np.sum(density_ratios)

    all_estimates = []
    for mat_src in empirical_mat_list_src:
        # Estimates is a 1-D array of estimates for each mat e.g.
        # each mat can correspond to a model's (n x 1) error matrix
        weighted_estimates = weighted_estimator(weights, mat_src)
        source_estimates = weighted_estimator(
            np.ones(solved.n_src) / solved.n_src, mat_src
        )

        all_estimates.append(
            SimpleNamespace(
                weighted=weighted_estimates,
                source=source_estimates,
            )
        )

    return SimpleNamespace(
        all_estimates=all_estimates,
        solved=solved,
        weights=weights,
    )


###########################################################################


def get_entropy(probas):
    return -np.sum(np.multiply(probas, np.log(probas + 1e-20)), axis=1)


def get_slices(probas, n_ent_bins=6):
    ln, ncl = probas.shape
    preds = np.argmax(probas, axis=1)
    pred_slices = np.full((ln, ncl), fill_value=-1, dtype="<i8")
    pred_slices[np.arange(ln), preds] = 1

    ent = get_entropy(probas)
    range_top = get_entropy(np.array([np.ones(ncl) / ncl]))[0]
    ent_bins = np.linspace(0, range_top, n_ent_bins + 1)
    bins_map = np.digitize(ent, bins=ent_bins, right=True) - 1
    ent_slices = np.full((ln, n_ent_bins), fill_value=-1, dtype="<i8")
    ent_slices[np.arange(ln), bins_map] = 1

    return np.concatenate([pred_slices, ent_slices], axis=1)
