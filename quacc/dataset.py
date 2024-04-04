import math
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

from quacc.legacy.environment import env
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


def save_dataset_stats(path, test_prot, L, V):
    test_prevs = [Ui.prevalence() for Ui in test_prot()]
    shifts = [qp.error.ae(L.prevalence(), Ui_prev) for Ui_prev in test_prevs]
    info = {
        "n_classes": L.n_classes,
        "n_train": len(L),
        "n_val": len(V),
        "train_prev": L.prevalence().tolist(),
        "val_prev": V.prevalence().tolist(),
        "test_prevs": [x.tolist() for x in test_prevs],
        "shifts": [x.tolist() for x in shifts],
        "sample_size": test_prot.sample_size,
        "num_samples": test_prot.total(),
    }
    save_json_file(path, info)


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
        train, U = qp.datasets.fetch_reviews(
            "imdb", tfidf=True, min_df=10, pickle=True
        ).train_test
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
        train = fetch_20newsgroups(
            subset="train", remove=("headers", "footers", "quotes")
        )
        test = fetch_20newsgroups(
            subset="test", remove=("headers", "footers", "quotes")
        )
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


class Dataset(DatasetProvider):
    def __init__(self, name, n_prevalences=9, prevs=None, target=None):
        self._name = name
        self._target = target

        self.all_train, self.test = self.alltrain_test(self._name, self._target)
        self.__resample_all_train()

        self.prevs = None
        self._n_prevs = n_prevalences
        self.__check_prevs(prevs)
        self.prevs = self.__build_prevs()

    def __resample_all_train(self):
        tr_counts, tr_ncl = self.all_train.counts(), self.all_train.n_classes
        _resample_prevs = np.full((tr_ncl,), fill_value=1.0 / tr_ncl)
        self.all_train = self.all_train.sampling(
            np.min(tr_counts) * tr_ncl,
            *_resample_prevs.tolist(),
            random_state=env._R_SEED,
        )

    def __check_prevs(self, prevs):
        try:
            iter(prevs)
        except TypeError:
            return

        if prevs is None or len(prevs) == 0:
            return

        def is_float_iterable(obj):
            try:
                it = iter(obj)
                return all([isinstance(o, float) for o in it])
            except TypeError:
                return False

        if not all([is_float_iterable(p) for p in prevs]):
            return

        if not all([len(p) == self.all_train.n_classes for p in prevs]):
            return

        if not all([sum(p) == 1.0 for p in prevs]):
            return

        self.prevs = np.unique(prevs, axis=0)

    def __build_prevs(self):
        if self.prevs is not None:
            return self.prevs

        dim = self.all_train.n_classes
        lspace = np.linspace(0.0, 1.0, num=self._n_prevs + 1, endpoint=False)[1:]
        mesh = np.array(np.meshgrid(*[lspace for _ in range(dim)])).T.reshape(-1, dim)
        mesh = np.around(mesh, decimals=4)
        mesh[np.where(np.around(mesh.sum(axis=1), decimals=4) == 0.9999), -1] += 0.0001
        mesh[np.where(np.around(mesh.sum(axis=1), decimals=4) == 1.0001), -1] -= 0.0001
        mesh = mesh[np.where(np.around(mesh.sum(axis=1), decimals=4) == 1.0)]
        return np.around(np.unique(mesh, axis=0), decimals=4)

    def __build_sample(
        self,
        p: np.ndarray,
        at_size: int,
    ):
        all_train_sampled = self.all_train.sampling(
            at_size, *(p[:-1]), random_state=env._R_SEED
        )
        train, validation = all_train_sampled.split_stratified(
            train_prop=TRAIN_VAL_PROP, random_state=env._R_SEED
        )
        return DatasetSample(train, validation, self.test)

    def get(self) -> List[DatasetSample]:
        at_size = min(
            math.floor(len(self.all_train) * (1.0 / self.all_train.n_classes) / p)
            for _prev in self.prevs
            for p in _prev
        )

        return [self.__build_sample(p, at_size) for p in self.prevs]

    def __call__(self):
        return self.get()

    @property
    def name(self):
        match (self._name, self._n_prevs):
            case (("rcv1" | "cifar10" | "cifar100"), 9):
                return f"{self._name}_{self._target}"
            case (("rcv1" | "cifar10" | "cifar100"), _):
                return f"{self._name}_{self._target}_{self._n_prevs}prevs"
            case (_, 9):
                return f"{self._name}"
            case (_, _):
                return f"{self._name}_{self._n_prevs}prevs"

    @property
    def nprevs(self):
        return self.prevs.shape[0]


# >>> fetch_rcv1().target_names
# array(['C11', 'C12', 'C13', 'C14', 'C15', 'C151', 'C1511', 'C152', 'C16',
#        'C17', 'C171', 'C172', 'C173', 'C174', 'C18', 'C181', 'C182',
#        'C183', 'C21', 'C22', 'C23', 'C24', 'C31', 'C311', 'C312', 'C313',
#        'C32', 'C33', 'C331', 'C34', 'C41', 'C411', 'C42', 'CCAT', 'E11',
#        'E12', 'E121', 'E13', 'E131', 'E132', 'E14', 'E141', 'E142',
#        'E143', 'E21', 'E211', 'E212', 'E31', 'E311', 'E312', 'E313',
#        'E41', 'E411', 'E51', 'E511', 'E512', 'E513', 'E61', 'E71', 'ECAT',
#        'G15', 'G151', 'G152', 'G153', 'G154', 'G155', 'G156', 'G157',
#        'G158', 'G159', 'GCAT', 'GCRIM', 'GDEF', 'GDIP', 'GDIS', 'GENT',
#        'GENV', 'GFAS', 'GHEA', 'GJOB', 'GMIL', 'GOBIT', 'GODD', 'GPOL',
#        'GPRO', 'GREL', 'GSCI', 'GSPO', 'GTOUR', 'GVIO', 'GVOTE', 'GWEA',
#        'GWELF', 'M11', 'M12', 'M13', 'M131', 'M132', 'M14', 'M141',
#        'M142', 'M143', 'MCAT'], dtype=object)


def rcv1_info():
    dataset = fetch_rcv1()
    n_train = 23149

    targets = []
    for target in ["CCAT", "MCAT", "GCAT"]:
        target_index = np.where(dataset.target_names == target)[0]
        train_t_prev = np.average(
            dataset.target[:n_train, target_index].toarray().flatten()
        )
        test_t_prev = np.average(
            dataset.target[n_train:, target_index].toarray().flatten()
        )
        d = Dataset(name="rcv1", target=target)()[0]
        targets.append(
            (
                target,
                {
                    "train": (1.0 - train_t_prev, train_t_prev),
                    "test": (1.0 - test_t_prev, test_t_prev),
                    "train_size": len(d.train),
                    "val_size": len(d.validation),
                    "test_size": len(d.test),
                },
            )
        )

    for n, d in targets:
        print(f"{n}:")
        for k, v in d.items():
            if isinstance(v, tuple):
                print(f"\t{k}: {v[0]:.4f}, {v[1]:.4f}")
            else:
                print(f"\t{k}: {v}")


def imdb_info():
    train, test = qp.datasets.fetch_reviews("imdb", tfidf=True, min_df=3).train_test

    train_t_prev = train.prevalence()
    test_t_prev = test.prevalence()
    dst = Dataset(name="imdb")()[0]
    d = {
        "train": (train_t_prev[0], train_t_prev[1]),
        "test": (test_t_prev[0], test_t_prev[1]),
        "train_size": len(dst.train),
        "val_size": len(dst.validation),
        "test_size": len(dst.test),
    }

    print("imdb:")
    for k, v in d.items():
        if isinstance(v, tuple):
            print(f"\t{k}: {v[0]:.4f}, {v[1]:.4f}")
        else:
            print(f"\t{k}: {v}")


if __name__ == "__main__":
    rcv1_info()
    imdb_info()
