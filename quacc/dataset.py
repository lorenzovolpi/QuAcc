import math
import os
import pickle
import tarfile
from typing import List, Tuple

import numpy as np
import quapy as qp
from quapy.data.base import LabelledCollection
from sklearn.conftest import fetch_rcv1
from sklearn.utils import Bunch

from quacc import utils
from quacc.environment import env

TRAIN_VAL_PROP = 0.5


def fetch_cifar10() -> Bunch:
    URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    data_home = utils.get_quacc_home()
    unzipped_path = data_home / "cifar-10-batches-py"
    if not unzipped_path.exists():
        downloaded_path = data_home / URL.split("/")[-1]
        utils.download_file(URL, downloaded_path)
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
    data_home = utils.get_quacc_home()
    unzipped_path = data_home / "cifar-100-python"
    if not unzipped_path.exists():
        downloaded_path = data_home / URL.split("/")[-1]
        utils.download_file(URL, downloaded_path)
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


class Dataset:
    def __init__(self, name, n_prevalences=9, prevs=None, target=None):
        self._name = name
        self._target = target

        self.prevs = None
        self.n_prevs = n_prevalences
        if prevs is not None:
            prevs = np.unique([p for p in prevs if p > 0.0 and p < 1.0])
            if prevs.shape[0] > 0:
                self.prevs = np.sort(prevs)
                self.n_prevs = self.prevs.shape[0]

    def __spambase(self):
        return qp.datasets.fetch_UCIDataset("spambase", verbose=False).train_test

    # provare min_df=5
    def __imdb(self):
        return qp.datasets.fetch_reviews("imdb", tfidf=True, min_df=3).train_test

    def __rcv1(self):
        n_train = 23149
        available_targets = ["CCAT", "GCAT", "MCAT"]

        if self._target is None or self._target not in available_targets:
            raise ValueError(f"Invalid target {self._target}")

        dataset = fetch_rcv1()
        target_index = np.where(dataset.target_names == self._target)[0]
        all_train_d = dataset.data[:n_train, :]
        test_d = dataset.data[n_train:, :]
        labels = dataset.target[:, target_index].toarray().flatten()
        all_train_l, test_l = labels[:n_train], labels[n_train:]
        all_train = LabelledCollection(all_train_d, all_train_l, classes=[0, 1])
        test = LabelledCollection(test_d, test_l, classes=[0, 1])

        return all_train, test

    def __cifar10(self):
        dataset = fetch_cifar10()
        available_targets: list = dataset.label_names

        if self._target is None or self._target not in available_targets:
            raise ValueError(f"Invalid target {self._target}")

        target_index = available_targets.index(self._target)
        all_train_d = dataset.train.data
        all_train_l = (dataset.train.labels == target_index).astype(int)
        test_d = dataset.test.data
        test_l = (dataset.test.labels == target_index).astype(int)
        all_train = LabelledCollection(all_train_d, all_train_l, classes=[0, 1])
        test = LabelledCollection(test_d, test_l, classes=[0, 1])

        return all_train, test

    def __cifar100(self):
        dataset = fetch_cifar100()
        available_targets: list = dataset.coarse_label_names

        if self._target is None or self._target not in available_targets:
            raise ValueError(f"Invalid target {self._target}")

        target_index = available_targets.index(self._target)
        all_train_d = dataset.train.data
        all_train_l = (dataset.train.coarse_labels == target_index).astype(int)
        test_d = dataset.test.data
        test_l = (dataset.test.coarse_labels == target_index).astype(int)
        all_train = LabelledCollection(all_train_d, all_train_l, classes=[0, 1])
        test = LabelledCollection(test_d, test_l, classes=[0, 1])

        return all_train, test

    def __train_test(self) -> Tuple[LabelledCollection, LabelledCollection]:
        all_train, test = {
            "spambase": self.__spambase,
            "imdb": self.__imdb,
            "rcv1": self.__rcv1,
            "cifar10": self.__cifar10,
            "cifar100": self.__cifar100,
        }[self._name]()

        return all_train, test

    def get_raw(self) -> DatasetSample:
        all_train, test = self.__train_test()

        train, val = all_train.split_stratified(
            train_prop=TRAIN_VAL_PROP, random_state=env._R_SEED
        )

        return DatasetSample(train, val, test)

    def get(self) -> List[DatasetSample]:
        all_train, test = self.__train_test()

        # resample all_train set to have (0.5, 0.5) prevalence
        at_positives = np.sum(all_train.y)
        all_train = all_train.sampling(
            min(at_positives, len(all_train) - at_positives) * 2,
            0.5,
            random_state=env._R_SEED,
        )

        # sample prevalences
        if self.prevs is not None:
            prevs = self.prevs
        else:
            prevs = np.linspace(0.0, 1.0, num=self.n_prevs + 1, endpoint=False)[1:]

        at_size = min(math.floor(len(all_train) * 0.5 / p) for p in prevs)
        datasets = []
        for p in 1.0 - prevs:
            all_train_sampled = all_train.sampling(at_size, p, random_state=env._R_SEED)
            train, validation = all_train_sampled.split_stratified(
                train_prop=TRAIN_VAL_PROP, random_state=env._R_SEED
            )
            datasets.append(DatasetSample(train, validation, test))

        return datasets

    def __call__(self):
        return self.get()

    @property
    def name(self):
        match (self._name, self.n_prevs):
            case (("rcv1" | "cifar10" | "cifar100"), 9):
                return f"{self._name}_{self._target}"
            case (("rcv1" | "cifar10" | "cifar100"), _):
                return f"{self._name}_{self._target}_{self.n_prevs}prevs"
            case (_, 9):
                return f"{self._name}"
            case (_, _):
                return f"{self._name}_{self.n_prevs}prevs"


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
    for target in range(103):
        train_t_prev = np.average(dataset.target[:n_train, target].toarray().flatten())
        test_t_prev = np.average(dataset.target[n_train:, target].toarray().flatten())
        targets.append(
            (
                dataset.target_names[target],
                {
                    "train": (1.0 - train_t_prev, train_t_prev),
                    "test": (1.0 - test_t_prev, test_t_prev),
                },
            )
        )

    targets.sort(key=lambda t: t[1]["train"][1])
    for n, d in targets:
        print(f"{n}:")
        for k, (fp, tp) in d.items():
            print(f"\t{k}: {fp:.4f}, {tp:.4f}")


if __name__ == "__main__":
    fetch_cifar100()
