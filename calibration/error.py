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
    bin_widths = bins[1:] - bins[:-1]
    binning = np.digitize(P[:, y], bins, right=True)

    bin_ce = np.zeros(n_bins)
    for b in np.unique(binning):
        b_idx = np.nonzero(binning == b)[0]
        bin_conf_err = (y[b_idx] == y_hat[b_idx]).mean() - P[b_idx, y].mean()
        bin_ce[b - 1] = bin_conf_err

    if norm == "l1":
        return np.sum(bin_widths * np.abs(bin_ce))
    elif norm == "l2":
        return np.sqrt(np.sum(bin_widths * bin_ce**2))
    elif norm == "inf":
        return np.max(bin_ce)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = rng.random(30)
    print(a)
    n_bins = 15
    bins = np.linspace(0, 1, n_bins + 1)
    binning = np.digitize(a, bins, right=True)
    print(binning)
