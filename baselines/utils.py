from sklearn import clone
from sklearn.base import BaseEstimator


def clone_fit(c_model: BaseEstimator, data, labels):
    c_model2 = clone(c_model)
    c_model2.fit(data, labels)
    return c_model2
