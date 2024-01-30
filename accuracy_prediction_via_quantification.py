import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import quapy as qp
from method.kdey import KDEyML, KDEyCS, KDEyHD
from quapy.protocol import APP
from quapy.method.aggregative import PACC, ACC, EMQ, PCC, CC, DMy

datasets = qp.datasets.UCI_DATASETS

# target = 'f1'
target = 'acc'

errors = []

# dataset_name = datasets[-2]
for dataset_name in datasets:
    if dataset_name in ['balance.2', 'acute.a', 'acute.b', 'iris.1']:
        continue
    train, test = qp.datasets.fetch_UCIDataset(dataset_name).train_test

    print(f'dataset name = {dataset_name}')
    print(f'#train = {len(train)}')
    print(f'#test = {len(test)}')

    cls = LogisticRegression()

    train, val = train.split_stratified(random_state=0)


    cls.fit(*train.Xy)
    y_val = val.labels
    y_hat_val = cls.predict(val.instances)

    for sample in APP(test, n_prevalences=11, repeats=1, sample_size=100, return_type='labelled_collection')():
        print('='*80)
        y_hat = cls.predict(sample.instances)
        y = sample.labels
        if target == 'acc':
            acc = (y_hat==y).mean()
        else:
            acc = f1_score(y, y_hat, zero_division=0)

        q = EMQ(cls)
        q.fit(train, fit_classifier=False)

        # q = EMQ(cls)
        # q.fit(train, val_split=val, fit_classifier=False)
        M_hat = ACC.getPteCondEstim(train.classes_, y_val, y_hat_val)
        M_true = ACC.getPteCondEstim(train.classes_, y, y_hat)
        p_hat = q.quantify(sample.instances)
        cont_table_hat = p_hat * M_hat

        tp = cont_table_hat[1,1]
        tn = cont_table_hat[0,0]
        fn = cont_table_hat[0,1]
        fp = cont_table_hat[1,0]

        if target == 'acc':
            acc_hat = (tp+tn)
        else:
            den = (2*tp + fn + fp)
            if den > 0:
                acc_hat = 2*tp / den
            else:
                acc_hat = 0

        error = abs(acc - acc_hat)
        errors.append(error)

        print('true_prev: ', sample.prevalence())
        print('estim_prev: ', p_hat)
        print('M-true:\n', M_true)
        print('M-hat:\n', M_hat)
        print('cont_table:\n', cont_table_hat)
        print(f'classifier accuracy={acc:.3f}')
        print(f'estimated accuracy={acc_hat:.3f}')
        print(f'estimation error={error:.4f}')

print('process end')
print('='*80)
print(f'mean error = {np.mean(errors)}')
print(f'std error = {np.std(errors)}')






