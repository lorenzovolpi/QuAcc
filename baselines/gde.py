import numpy as np


def get_score(pred1, pred2):
    return np.mean(pred1 == pred2)
