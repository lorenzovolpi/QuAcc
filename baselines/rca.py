import numpy as np


def get_score(pred1, pred2, labels):
    return np.mean((pred1 == labels).astype(int) - (pred2 == labels).astype(int))
