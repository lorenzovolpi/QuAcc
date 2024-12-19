import numpy as np


def calibration_error(P, y, n_bins=15, adaptive_binning=False, norm="l1"):
    """
    Computes the calibration from the posteriors obtained from a classification model given the true labels.

    :param P: posteriors obtained from the model on a dataset
    :param y: true labels of the dataset
    :param n_bins: number of bins used to compute the error; default value=15
    :param adataptive_binning: if true all bins contain the same number of datapoints, otherwise all bins have the same width; default value=False
    :param norm: norm used to compute the error; available values are 'l1', 'l2' and 'inf'; default value='l1'
    :return: calibration error
    """
    y_hat = np.argmax(P, axis=1)
    bins = np.linspace(0, 1, n_bins + 1)
    binning = np.digitize(P[:, y], bins, right=True)

    bin_ce = []
    for b in np.arange(n_bins + 1)[1:]:
        b_idx = np.nonzero(binning == b)[0]
        bin_conf_err = (y[b_idx] == y_hat[b_idx]).mean() - P[b_idx, y].mean()
