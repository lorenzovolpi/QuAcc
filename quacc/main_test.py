import logging
from logging.handlers import QueueHandler
from multiprocessing import Manager, Queue
from threading import Thread
from time import sleep, time

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from quapy.protocol import APP
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score

from baselines.mandoline import estimate_performance
from quacc.dataset import Dataset
from quacc.logger import logger, logger_manager, setup_logger, setup_worker_logger


def test_lr():
    d = Dataset(name="rcv1", target="CCAT", n_prevalences=1).get_raw()

    classifier = LogisticRegression()
    classifier.fit(*d.train.Xy)

    val, _ = d.validation.split_stratified(0.5, random_state=0)
    val_X, val_y = val.X, val.y
    val_probs = classifier.predict_proba(val_X)

    reg_X = sp.hstack([val_X, val_probs])
    reg_y = val_probs[np.arange(val_probs.shape[0]), val_y]
    reg = LinearRegression()
    reg.fit(reg_X, reg_y)

    _test_num = 10000
    test_X = d.test.X[:_test_num, :]
    test_probs = classifier.predict_proba(test_X)
    test_reg_X = sp.hstack([test_X, test_probs])
    reg_pred = reg.predict(test_reg_X)

    def threshold(pred):
        # return np.mean(
        #     (reg.predict(test_reg_X) >= pred)
        #     == (
        #         test_probs[np.arange(_test_num), d.test.y[:_test_num]] == np.max(test_probs, axis=1)
        #     )
        # )
        return np.mean(
            (reg.predict(test_reg_X) >= pred)
            == (np.argmax(test_probs, axis=1) == d.test.y[:_test_num])
        )

    max_p, max_acc = 0, 0
    for p in reg_pred:
        acc = threshold(p)
        if acc > max_acc:
            max_acc = acc
            max_p = p

    print(f"{max_p = }, {max_acc = }")
    reg_pred = reg_pred - max_p + 0.5
    print(reg_pred)
    print(np.mean(reg_pred >= 0.5))
    print(np.mean(np.argmax(test_probs, axis=1) == d.test.y[:_test_num]))


def entropy(probas):
    return -np.sum(np.multiply(probas, np.log(probas + 1e-20)), axis=1)


def get_slices(probas):
    ln, ncl = probas.shape
    preds = np.argmax(probas, axis=1)
    pred_slices = np.full((ln, ncl), fill_value=-1, dtype="<i8")
    pred_slices[np.arange(ln), preds] = 1

    ent = entropy(probas)
    n_bins = 10
    range_top = entropy(np.array([np.ones(ncl) / ncl]))[0]
    bins = np.linspace(0, range_top, n_bins + 1)
    bin_map = np.digitize(ent, bins=bins, right=True) - 1
    ent_slices = np.full((ln, n_bins), fill_value=-1, dtype="<i8")
    ent_slices[np.arange(ln), bin_map] = 1

    return np.concatenate([pred_slices, ent_slices], axis=1)


def test_mandoline():
    d = Dataset(name="cifar10", target="dog", n_prevalences=1).get_raw()

    tstart = time()
    classifier = LogisticRegression()
    classifier.fit(*d.train.Xy)

    val_probs = classifier.predict_proba(d.validation.X)
    val_preds = np.argmax(val_probs, axis=1)
    D_val = get_slices(val_probs)
    emprical_mat_list_val = (1.0 * (val_preds == d.validation.y))[:, np.newaxis]

    protocol = APP(
        d.test,
        sample_size=1000,
        n_prevalences=21,
        repeats=100,
        return_type="labelled_collection",
    )
    res = []
    for test in protocol():
        test_probs = classifier.predict_proba(test.X)
        test_preds = np.argmax(test_probs, axis=1)
        D_test = get_slices(test_probs)
        wp = estimate_performance(D_val, D_test, None, emprical_mat_list_val)
        score = wp.all_estimates[0].weighted[0]
        res.append(abs(score - accuracy_score(test.y, test_preds)))
        print(score)
    res = np.array(res).reshape((21, 100))
    print(res.mean(axis=1))
    print(f"time: {time() - tstart}s")


def joblib_queue():
    def worker(q: Queue, i):
        setup_worker_logger(q)
        log = logger()
        log.info(i)
        sleep(2)
        print(f"worker {i}")

    setup_logger()
    log = logger()
    log.info("start")
    Parallel(n_jobs=5)(delayed(worker)(logger_manager().q, i) for i in range(5))
    log.info("end")
    logger_manager().close()


if __name__ == "__main__":
    joblib_queue()
