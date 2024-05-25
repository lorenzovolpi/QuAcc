import numpy as np
import scipy


def get_posteriors_from_h(h, X):
    if hasattr(h, "predict_proba"):
        P = h.predict_proba(X)
    else:
        n_classes = len(h.classes_)
        dec_scores = h.decision_function(X)
        if n_classes == 1:
            dec_scores = np.vstack([-dec_scores, dec_scores]).T
        P = scipy.special.softmax(dec_scores, axis=1)
    return P


def max_conf(P, keepdims=False):
    mc = P.max(axis=1, keepdims=keepdims)
    return mc


def neg_entropy(P, keepdims=False):
    ne = scipy.stats.entropy(P, axis=1)
    if keepdims:
        ne = ne.reshape(-1, 1)
    return ne


def smooth(prevalences, epsilon=1e-5):
    """
    Smooths a prevalence vector.

    :param prevalences: np.ndarray
    :param epsilon: float, a small quantity (default 1E-5)
    :return: smoothed prevalence vector
    """
    prevalences = prevalences + epsilon
    prevalences /= prevalences.sum()
    return prevalences
