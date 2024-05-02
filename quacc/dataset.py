import os
import pickle
import tarfile
from typing import List

import numpy as np
import quapy as qp
from quapy.data.base import LabelledCollection
from quapy.data.datasets import fetch_lequa2022, fetch_UCIMulticlassLabelledCollection
from sklearn.datasets import fetch_20newsgroups, fetch_rcv1
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import Bunch

import quacc as qc
from quacc.utils import commons
from quacc.utils.commons import save_json_file

TRAIN_VAL_PROP = 0.5


def fetch_cifar10() -> Bunch:
    URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    data_home = commons.get_quacc_home()
    unzipped_path = data_home / "cifar-10-batches-py"
    if not unzipped_path.exists():
        downloaded_path = data_home / URL.split("/")[-1]
        commons.download_file(URL, downloaded_path)
        with tarfile.open(downloaded_path) as f:
            f.extractall(data_home)
        os.remove(downloaded_path)

    datas = []
    data_names = sorted([f for f in os.listdir(unzipped_path) if f.startswith("data")])
    for f in data_names:
        with open(unzipped_path / f, "rb") as file:
            datas.append(pickle.load(file, encoding="bytes"))

    tests = []
    test_names = sorted([f for f in os.listdir(unzipped_path) if f.startswith("test")])
    for f in test_names:
        with open(unzipped_path / f, "rb") as file:
            tests.append(pickle.load(file, encoding="bytes"))

    with open(unzipped_path / "batches.meta", "rb") as file:
        meta = pickle.load(file, encoding="bytes")

    return Bunch(
        train=Bunch(
            data=np.concatenate([d[b"data"] for d in datas], axis=0),
            labels=np.concatenate([d[b"labels"] for d in datas]),
        ),
        test=Bunch(
            data=np.concatenate([d[b"data"] for d in tests], axis=0),
            labels=np.concatenate([d[b"labels"] for d in tests]),
        ),
        label_names=[cs.decode("utf-8") for cs in meta[b"label_names"]],
    )


def fetch_cifar100():
    URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    data_home = commons.get_quacc_home()
    unzipped_path = data_home / "cifar-100-python"
    if not unzipped_path.exists():
        downloaded_path = data_home / URL.split("/")[-1]
        commons.download_file(URL, downloaded_path)
        with tarfile.open(downloaded_path) as f:
            f.extractall(data_home)
        os.remove(downloaded_path)

    with open(unzipped_path / "train", "rb") as file:
        train_d = pickle.load(file, encoding="bytes")

    with open(unzipped_path / "test", "rb") as file:
        test_d = pickle.load(file, encoding="bytes")

    with open(unzipped_path / "meta", "rb") as file:
        meta_d = pickle.load(file, encoding="bytes")

    train_bunch = Bunch(
        data=train_d[b"data"],
        fine_labels=np.array(train_d[b"fine_labels"]),
        coarse_labels=np.array(train_d[b"coarse_labels"]),
    )

    test_bunch = Bunch(
        data=test_d[b"data"],
        fine_labels=np.array(test_d[b"fine_labels"]),
        coarse_labels=np.array(test_d[b"coarse_labels"]),
    )

    return Bunch(
        train=train_bunch,
        test=test_bunch,
        fine_label_names=meta_d[b"fine_label_names"],
        coarse_label_names=meta_d[b"coarse_label_names"],
    )


class DatasetSample:
    def __init__(
        self,
        train: LabelledCollection,
        validation: LabelledCollection,
        test: LabelledCollection,
    ):
        self.train = train
        self.validation = validation
        self.test = test

    @property
    def train_prev(self):
        return self.train.prevalence()

    @property
    def validation_prev(self):
        return self.validation.prevalence()

    @property
    def prevs(self):
        return {"train": self.train_prev, "validation": self.validation_prev}


class DatasetProvider:
    @classmethod
    def _split_train(cls, train: LabelledCollection):
        return train.split_stratified(0.5, random_state=0)

    @classmethod
    def _split_whole(cls, dataset: LabelledCollection):
        train, U = dataset.split_stratified(train_prop=0.66, random_state=0)
        T, V = train.split_stratified(train_prop=0.5, random_state=0)
        return T, V, U

    @classmethod
    def spambase(cls):
        train, U = qp.datasets.fetch_UCIDataset("spambase", verbose=False).train_test
        T, V = cls._split_train(train)
        return T, V, U

    @classmethod
    def imdb(cls):
        train, U = qp.datasets.fetch_reviews("imdb", tfidf=True, min_df=10, pickle=True).train_test
        T, V = cls._split_train(train)
        return T, V, U

    @classmethod
    def rcv1(cls, target):
        training = fetch_rcv1(subset="train")
        test = fetch_rcv1(subset="test")

        available_targets = ["CCAT", "GCAT", "MCAT"]
        if target is None or target not in available_targets:
            raise ValueError(f"Invalid target {target}")

        class_names = training.target_names.tolist()
        class_idx = class_names.index(target)
        tr_labels = training.target[:, class_idx].toarray().flatten()
        te_labels = test.target[:, class_idx].toarray().flatten()
        tr = LabelledCollection(training.data, tr_labels)
        U = LabelledCollection(test.data, te_labels)
        T, V = cls._split_train(tr)
        return T, V, U

    @classmethod
    def cifar10(cls, target):
        dataset = fetch_cifar10()
        available_targets: list = dataset.label_names

        if target is None or target not in available_targets:
            raise ValueError(f"Invalid target {target}")

        target_idx = available_targets.index(target)
        train_d = dataset.train.data
        train_l = (dataset.train.labels == target_idx).astype(int)
        test_d = dataset.test.data
        test_l = (dataset.test.labels == target_idx).astype(int)
        train = LabelledCollection(train_d, train_l, classes=[0, 1])
        U = LabelledCollection(test_d, test_l, classes=[0, 1])
        T, V = cls._split_train(train)

        return T, V, U

    @classmethod
    def cifar100(cls, target):
        dataset = fetch_cifar100()
        available_targets: list = dataset.coarse_label_names

        if target is None or target not in available_targets:
            raise ValueError(f"Invalid target {target}")

        target_index = available_targets.index(target)
        train_d = dataset.train.data
        train_l = (dataset.train.coarse_labels == target_index).astype(int)
        test_d = dataset.test.data
        test_l = (dataset.test.coarse_labels == target_index).astype(int)
        train = LabelledCollection(train_d, train_l, classes=[0, 1])
        U = LabelledCollection(test_d, test_l, classes=[0, 1])
        T, V = cls._split_train(train)

        return T, V, U

    @classmethod
    def twitter(cls, dataset_name):
        data = qp.datasets.fetch_twitter(dataset_name, min_df=3, pickle=True)
        T, V = cls._split_train(data.training)
        U = data.test
        return T, V, U

    @classmethod
    def uci_multiclass(cls, dataset_name):
        dataset = fetch_UCIMulticlassLabelledCollection(dataset_name)
        return cls._split_whole(dataset)

    @classmethod
    def news20(cls):
        train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
        test = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
        tfidf = TfidfVectorizer(min_df=5, sublinear_tf=True)
        Xtr = tfidf.fit_transform(train.data)
        Xte = tfidf.transform((test.data))
        train = LabelledCollection(instances=Xtr, labels=train.target)
        U = LabelledCollection(instances=Xte, labels=test.target)
        T, V = cls._split_train(train)
        return T, V, U

    @classmethod
    def t1b_lequa2022(cls):
        dataset, _, _ = fetch_lequa2022(task="T1B")
        return cls._split_whole(dataset)
