import numpy as np
import scipy.special
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import quapy as qp
from quapy.protocol import APP
from quapy.method.aggregative import PACC, ACC, EMQ, PCC, CC, DMy, T50, MS2, KDEyML, KDEyCS, KDEyHD
from sklearn import clone
import quapy.functional as F

# datasets = qp.datasets.UCI_DATASETS
datasets = ['imdb']

# target = 'f1'
target = 'acc'

errors = []

def method_1(cls, train, val, sample, y=None, y_hat=None):
    """
    Converts a misclassification matrix computed in validation (i.e., in the train distribution P) into
    the corresponding equivalent misclassification matrix in test (i.e., in the test distribution Q)
    by relying on the PPS assumptions.

    :return: tuple (tn, fn, fp, tp,) of floats in [0,1] summing up to 1
    """

    y_val = val.labels
    y_hat_val = cls.predict(val.instances)

    # q = EMQ(LogisticRegression(class_weight='balanced'))
    # q.fit(val, fit_classifier=True)
    q = EMQ(cls)
    q.fit(train, fit_classifier=False)


    # q = KDEyML(cls)
    # q.fit(train, val_split=val, fit_classifier=False)
    M_hat = ACC.getPteCondEstim(train.classes_, y_val, y_hat_val)
    M_true = ACC.getPteCondEstim(train.classes_, y, y_hat)
    p_hat = q.quantify(sample.instances)
    cont_table_hat = p_hat * M_hat
    # cont_table_hat = np.clip(cont_table_hat, 0, 1)
    # cont_table_hat = cont_table_hat / cont_table_hat.sum()

    print('true_prev: ', sample.prevalence())
    print('estim_prev: ', p_hat)
    print('M-true:\n', M_true)
    print('M-hat:\n', M_hat)
    print('cont_table:\n', cont_table_hat)
    print('cont_table Sum :\n', cont_table_hat.sum())

    tp = cont_table_hat[1, 1]
    tn = cont_table_hat[0, 0]
    fn = cont_table_hat[0, 1]
    fp = cont_table_hat[1, 0]

    return tn, fn, fp, tp


def method_2(cls, train, val, sample, y=None, y_hat=None):
    """
    Assume P and Q are the training and test distributions
    Solves the following system of linear equations:
    tp + fp = CC (the classify & count estimate, observed)
    fn + tp = Q(Y=1) (this is not observed but is estimated via quantification)
    tp + fp + fn + tn = 1 (trivial)

    There are 4 unknowns and 3 equations. The fourth required one is established
    by assuming that the PPS conditions hold, i.e., that P(X|Y)=Q(X|Y); note that
    this implies P(hatY|Y)=Q(hatY|Y) if hatY is computed by any measurable function.
    In particular, we consider that the tpr in P (estimated via validation, hereafter tpr) and
    in Q (unknown, hereafter tpr_Q) should
    be the same. This means:
    tpr = tpr_Q = tp / (tp + fn)
    after some manipulation:
    tp (tpr-1) + fn (tpr) = 0 <-- our last equation

    Note that the last equation relies on the estimate tpr. It is likely that, the more
    positives we have, the more reliable this estimate is. This suggests that, in cases
    in which we have more negatives in the validation set than positives, it might be
    convenient to resort to the true negative rate (tnr) instead. This gives rise to
    the alternative fourth equation:
    tn (tnr-1) + fp (tnr) = 0

    :return: tuple (tn, fn, fp, tp,) of floats in [0,1] summing up to 1
    """

    y_val = val.labels
    y_hat_val = cls.predict(val.instances)

    q = ACC(cls)
    q.fit(train, val_split=val, fit_classifier=False)
    p_hat = q.quantify(sample.instances)
    pos_prev = p_hat[1]
    # pos_prev = sample.prevalence()[1]

    cc = CC(cls)
    cc.fit(train, fit_classifier=False)
    cc_prev = cc.quantify(sample.instances)[1]

    M_hat = ACC.getPteCondEstim(train.classes_, y_val, y_hat_val)
    M_true = ACC.getPteCondEstim(train.classes_, y, y_hat)
    cont_table_true = sample.prevalence() * M_true

    if val.prevalence()[1] > 0.5:

        # in this case, the tpr might be a more reliable estimate than tnr
        tpr_hat = M_hat[1, 1]

        A = np.asarray([
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 1, 1],
            [0, tpr_hat, 0, tpr_hat - 1]
        ])

    else:

        # in this case, the tnr might be a more reliable estimate than tpr
        tnr_hat = M_hat[0, 0]

        A = np.asarray([
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 1, 1],
            [tnr_hat-1, 0, tnr_hat, 0]
        ])

    b = np.asarray(
        [cc_prev, pos_prev, 1, 0]
    )

    tn, fn, fp, tp = np.linalg.solve(A, b)

    cont_table_estim = np.asarray([
        [tn, fn],
        [fp, tp]
    ])

    # if (cont_table_estim < 0).any() or (cont_table_estim>1).any():
    #     cont_table_estim = scipy.special.softmax(cont_table_estim)

    print('true_prev: ', sample.prevalence())
    print('estim_prev: ', p_hat)
    print('true_cont_table:\n', cont_table_true)
    print('estim_cont_table:\n', cont_table_estim)
    # print('true_tpr', M_true[1,1])
    # print('estim_tpr', tpr_hat)


    return tn, fn, fp, tp


def method_3(cls, train, val, sample, y=None, y_hat=None):
    """
    This is just method 2 but without involving any quapy's quantifier.

    :return: tuple (tn, fn, fp, tp,) of floats in [0,1] summing up to 1
    """

    classes = val.classes_
    y_val = val.labels
    y_hat_val = cls.predict(val.instances)
    M_hat = ACC.getPteCondEstim(classes, y_val, y_hat_val)
    y_hat_test = cls.predict(sample.instances)
    pos_prev_cc = F.prevalence_from_labels(y_hat_test, classes)[1]
    tpr_hat = M_hat[1,1]
    fpr_hat = M_hat[1,0]
    tnr_hat = M_hat[0,0]
    pos_prev_test_hat = (pos_prev_cc - fpr_hat) / (tpr_hat - fpr_hat)
    pos_prev_test_hat = np.clip(pos_prev_test_hat, 0, 1)
    pos_prev_val = val.prevalence()[1]

    if pos_prev_val > 0.5:
        # in this case, the tpr might be a more reliable estimate than tnr
        A = np.asarray([
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 1, 1],
            [0, tpr_hat, 0, tpr_hat - 1]
        ])
    else:
        # in this case, the tnr might be a more reliable estimate than tpr
        A = np.asarray([
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 1, 1],
            [tnr_hat-1, 0, tnr_hat, 0]
        ])

    b = np.asarray(
        [pos_prev_cc, pos_prev_test_hat, 1, 0]
    )

    tn, fn, fp, tp = np.linalg.solve(A, b)

    return tn, fn, fp, tp


def cls_eval_from_counters(tn, fn, fp, tp):
    if target == 'acc':
        acc_hat = (tp + tn)
    else:
        den = (2 * tp + fn + fp)
        if den > 0:
            acc_hat = 2 * tp / den
        else:
            acc_hat = 0
    return acc_hat


def cls_eval_from_labels(y, y_hat):
    if target == 'acc':
        acc = (y_hat == y).mean()
    else:
        acc = f1_score(y, y_hat, zero_division=0)
    return acc


for dataset_name in datasets:

    train_orig, test = qp.datasets.fetch_reviews(dataset_name, tfidf=True, min_df=10).train_test

    train_prot = APP(train_orig, n_prevalences=11, repeats=1, return_type='labelled_collection', random_state=0, sample_size=10000)
    for train in train_prot():
        if np.product(train.prevalence()) == 0:
            # skip experiments with no positives or no negatives in training
            continue

        cls = LogisticRegression(class_weight='balanced')

        train, val = train.split_stratified(train_prop=0.5, random_state=0)

        print(f'dataset name = {dataset_name}')
        print(f'#train = {len(train)}, prev={F.strprev(train.prevalence())}')
        print(f'#val = {len(val)}, prev={F.strprev(val.prevalence())}')
        print(f'#test = {len(test)}, prev={F.strprev(test.prevalence())}')

        cls.fit(*train.Xy)

        for sample in APP(test, n_prevalences=21, repeats=10, sample_size=1000, return_type='labelled_collection')():
            print('='*80)
            y_hat = cls.predict(sample.instances)
            y = sample.labels
            acc_true = cls_eval_from_labels(y, y_hat)

            tn, fn, fp, tp = method_3(cls, train, val, sample, y, y_hat)

            acc_hat = cls_eval_from_counters(tn, fn, fp, tp)

            error = abs(acc_true - acc_hat)
            errors.append(error)

            print(f'classifier accuracy={acc_true:.3f}')
            print(f'estimated accuracy={acc_hat:.3f}')
            print(f'estimation error={error:.4f}')

print('process end')
print('='*80)
print(f'mean error = {np.mean(errors)}')
print(f'std error = {np.std(errors)}')






