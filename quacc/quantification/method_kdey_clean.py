from typing import Union, Callable
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import KernelDensity
from quapy.data import LabelledCollection
from quapy.method.aggregative import AggregativeProbabilisticQuantifier, cross_generate_predictions
import quapy as qp


class KDEy(AggregativeProbabilisticQuantifier):

    def __init__(self, classifier: BaseEstimator, val_split=10, bandwidth=0.1, n_jobs=None, random_state=0):
        self.classifier = classifier
        self.val_split = val_split
        self.bandwidth = bandwidth
        self.n_jobs = n_jobs
        self.random_state = random_state

    def get_kde_function(self, posteriors):
        return KernelDensity(bandwidth=self.bandwidth).fit(posteriors)

    def pdf(self, kde, posteriors):
        return np.exp(kde.score_samples(posteriors))

    def fit(self, data: LabelledCollection, fit_classifier=True, val_split: Union[float, LabelledCollection] = None):
        """

        :param data: the training set
        :param fit_classifier: set to False to bypass the training (the learner is assumed to be already fit)
        :param val_split: either a float in (0,1) indicating the proportion of training instances to use for
         validation (e.g., 0.3 for using 30% of the training set as validation data), or a LabelledCollection
         indicating the validation set itself, or an int indicating the number k of folds to be used in kFCV
         to estimate the parameters
        """
        if val_split is None:
            val_split = self.val_split

        with qp.util.temp_seed(self.random_state):
            self.classifier, y, posteriors, classes, class_count = cross_generate_predictions(
                data, self.classifier, val_split, probabilistic=True, fit_classifier=fit_classifier, n_jobs=self.n_jobs
            )
            self.val_densities = [self.get_kde_function(posteriors[y == cat]) for cat in range(data.n_classes)]

        return self

    def aggregate(self, posteriors: np.ndarray):
        """
        Searches for the mixture model parameter (the sought prevalence values) that yields a validation distribution
        (the mixture) that best matches the test distribution, in terms of the divergence measure of choice.

        :param instances: instances in the sample
        :return: a vector of class prevalence estimates
        """
        eps = 1e-10
        np.random.RandomState(self.random_state)
        n_classes = len(self.val_densities)
        test_densities = [self.pdf(kde_i, posteriors) for kde_i in self.val_densities]

        def neg_loglikelihood(prev):
            test_mixture_likelihood = sum(prev_i * dens_i for prev_i, dens_i in zip (prev, test_densities))
            test_loglikelihood = np.log(test_mixture_likelihood + eps)
            return -np.sum(test_loglikelihood)

        return optim_minimize(neg_loglikelihood, n_classes)


def optim_minimize(loss, n_classes):
    """
    Searches for the optimal prevalence values, i.e., an `n_classes`-dimensional vector of the (`n_classes`-1)-simplex
    that yields the smallest lost. This optimization is carried out by means of a constrained search using scipy's
    SLSQP routine.

    :param loss: (callable) the function to minimize
    :param n_classes: (int) the number of classes, i.e., the dimensionality of the prevalence vector
    :return: (ndarray) the best prevalence vector found
    """
    from scipy import optimize

    # the initial point is set as the uniform distribution
    uniform_distribution = np.full(fill_value=1 / n_classes, shape=(n_classes,))

    # solutions are bounded to those contained in the unit-simplex
    bounds = tuple((0, 1) for _ in range(n_classes))  # values in [0,1]
    constraints = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})  # values summing up to 1
    r = optimize.minimize(loss, x0=uniform_distribution, method='SLSQP', bounds=bounds, constraints=constraints)
    return r.x