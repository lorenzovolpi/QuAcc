import contextlib
import os
from collections import defaultdict
from typing import Callable, List

import datasets
import numpy as np
import pandas as pd
import quapy as qp
import torch
import torch.nn as nn
import torch.utils.data
from datasets import load_dataset
from quapy.data.base import LabelledCollection
from quapy.method.aggregative import SLD
from quapy.protocol import UPP
from scipy.sparse import issparse
from sklearn.kernel_ridge import KernelRidge as KRR
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier as MLP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    get_scheduler,
)

import quacc as qc
from quacc.error import vanilla_acc
from quacc.experiments.util import split_validation
from quacc.models.cont_table import QuAcc1xN2, QuAcc1xNp1, QuAccNxN
from quacc.models.direct import DoC
from quacc.models.model_selection import GridSearchCAP
from quacc.models.regression import ReQua
from quacc.models.utils import get_posteriors_from_h

NUM_SAMPLES = 100
SAMPLE_SIZE = 500
RANDOM_STATE = 4125

qp.environ["_R_SEED"] = RANDOM_STATE
qp.environ["SAMPLE_SIZE"] = SAMPLE_SIZE


def softmax(logits: torch.Tensor) -> np.ndarray:
    return nn.functional.softmax(logits, dim=-1).numpy()


class TorchLabelledCollection(LabelledCollection):
    def __init__(self, instances, labels, attention_mask, classes=None):
        self.attention_mask = attention_mask
        super().__init__(instances, labels, classes)

    def sampling_from_index(self, index):
        documents = self.instances[index]
        labels = self.labels[index]
        attention_mask = self.attention_mask[index]
        return TorchLabelledCollection(documents, labels, attention_mask, classes=self.classes_)


class TorchDataset(torch.utils.data.Dataset):
    DATA_MAPPING = {
        "X": "input_ids",
        "y": "labels",
        "mask": "attention_mask",
    }

    def __init__(self, X, y, attention_mask, data_mapping=None, has_labels=True):
        self.X = X
        self.y = y
        self.attention_mask = attention_mask
        self.data_mapping = self.DATA_MAPPING if data_mapping is None else data_mapping
        self.has_labels = has_labels

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        res = {}
        if issparse(self.X):
            res["X"] = self.X[index, :].toarray()
        else:
            res["X"] = self.X[index, :]

        if self.has_labels:
            res["y"] = self.y[index]

        res["mask"] = self.attention_mask[index, :]

        return {self.data_mapping[k]: v for k, v in res.items()}

    @classmethod
    def from_lc(cls, data: TorchLabelledCollection, data_mapping=None):
        return TorchDataset(*data.Xy, data.attention_mask, data_mapping=data_mapping)

    @classmethod
    def from_X(cls, data, attention_mask, data_mapping=None):
        return TorchDataset(data, None, attention_mask, data_mapping=data_mapping, has_labels=False)


class LargeModel:
    def fit(self, train: LabelledCollection, dataset_name: str): ...

    def predict_proba(self, test) -> np.ndarray: ...

    def get_model_path(self, dataset_name):
        return os.path.join(qc.env["OUT_DIR"], "models", f"{self.name}_on_{dataset_name}")

    def predict_from_proba(self, proba: np.ndarray) -> np.ndarray:
        return np.argmax(proba, axis=-1)

    def predict(self, test) -> np.ndarray:
        return self.predict_from_proba(self.predict_proba(test))


class DistilBert(LargeModel):
    def __init__(self, learning_rate=1e-5, batch_size=64, epochs=3, seed=RANDOM_STATE):
        self.name = "distilbert-base-uncased"
        self.tokenizer_name = self.name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = epochs
        self.seed = seed
        self.epoch = 0

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding="max_length")
        self.model: DistilBertForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(self.name)

    def get_model_path(self, dataset_name):
        return os.path.join(qc.env["OUT_DIR"], "models", f"{self.name}_on_{dataset_name}.tar")

    def checkpoint(self, dataset_name):
        path = self.get_model_path(dataset_name)
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_from_checkpoint(self, dataset_name):
        path = self.get_model_path(dataset_name)
        if os.path.exists(path):
            _checkpoint = torch.load(path)
            self.model.load_state_dict(_checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(_checkpoint["optimizer_state_dict"])
            self.epoch = _checkpoint["epoch"]

    def fit(self, train: TorchLabelledCollection, dataset_name: str, verbose=True):
        self.classes_ = train.classes_
        train_dl = DataLoader(
            TorchDataset.from_lc(train),
            batch_size=self.batch_size,
        )
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        num_training_steps = self.num_epochs * len(train_dl)

        self.load_from_checkpoint(dataset_name)

        if self.num_epochs == self.epoch:
            return self

        lr_scheduler = get_scheduler(
            name="linear", optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(device)

        if verbose:
            progress_bar = tqdm(range(num_training_steps))

        self.model.train()
        for epoch in range(self.num_epochs - self.epoch):
            self.epoch = epoch + 1
            for batch in train_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                self.optimizer.step()
                lr_scheduler.step()
                self.optimizer.zero_grad()
                if verbose:
                    progress_bar.update(1)

        self.checkpoint(dataset_name)

        return self

    def predict_proba(self, test, attention_mask, verbose=False) -> np.ndarray:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        test_dl = DataLoader(
            TorchDataset.from_X(test, attention_mask),
            batch_size=self.batch_size,
        )
        self.model.to(device)

        self.model.eval()
        y_probs = []
        if verbose:
            progress_bar = tqdm(range(len(test_dl)))
        with torch.no_grad():
            for batch in test_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                y_probs.append(softmax(outputs.logits.cpu()))
                if verbose:
                    progress_bar.update(1)

        return np.vstack(y_probs)


def preprocess_data(
    dataset, name, tokenizer, columns, length: int | None = None, seed=RANDOM_STATE
) -> datasets.Dataset:
    def tokenize(datapoint):
        sentences = [datapoint[c] for c in columns]
        return tokenizer(*sentences, truncation=True)

    d = dataset[name].shuffle(seed=seed)
    if length is not None:
        d = d.select(np.arange(length))
    d = d.map(tokenize, batched=True)
    return d


def from_hf_dataset(
    dataset: datasets.Dataset, collator: Callable, remove_columns: str | List[str] | None = None
) -> TorchLabelledCollection:
    if remove_columns is not None:
        dataset = dataset.remove_columns(remove_columns)
    ds = next(iter(DataLoader(dataset, collate_fn=collator, batch_size=len(dataset))))
    # print(ds["input_ids"].shape, ds["labels"].shape, ds["attention_mask"].shape)
    return TorchLabelledCollection(instances=ds["input_ids"], labels=ds["labels"], attention_mask=ds["attention_mask"])


def mlp():
    MLP((100, 15), activation="logistic", solver="adam")


def sld():
    _sld = SLD(LogisticRegression(), val_split=5)
    _sld.SUPPRESS_WARNINGS = True
    return _sld


def main():
    BASE = False
    OPT_N2 = False
    OPT_NN = False
    REQUA = True

    model = DistilBert()

    dataset_map = {
        # "imdb": (["text"], 25000),
        # "rotten_tomatoes": (["text"], 8530),
        "amazon_polarity": (["title", "content"], 25000),
    }

    results = defaultdict(lambda: [])
    for dataset_name, (text_columns, TRAIN_LENGTH) in dataset_map.items():
        print("-" * 10, dataset_name, "-" * 10)
        dataset = load_dataset(dataset_name)

        train_vec = preprocess_data(dataset, "train", model.tokenizer, text_columns, length=TRAIN_LENGTH)
        test_vec = preprocess_data(dataset, "test", model.tokenizer, text_columns)

        # print(f"train_vec len: {len(train_vec['input_ids'])}")
        # print(f"max train_vec lens: {max([len(le) for le in train_vec['input_ids']])}")
        # print(f"test_vec len: {len(test_vec['input_ids'])}")
        # print(f"max test_vec lens: {max([len(le) for le in test_vec['input_ids']])}")

        train = from_hf_dataset(train_vec, model.data_collator, remove_columns=text_columns)
        U = from_hf_dataset(test_vec, model.data_collator, remove_columns=text_columns)
        L, V = train.split_stratified(train_prop=0.5, random_state=RANDOM_STATE)

        model.fit(L, dataset_name)

        test_prot = UPP(U, sample_size=SAMPLE_SIZE, repeats=NUM_SAMPLES, return_type="labelled_collection")

        V1, V2_prot = split_validation(V)
        # print(f"V1 shape: {V1.X.shape}")
        # for i, v2 in enumerate(V2_prot()):
        #     print(f"v2_prot#{i} shape: {v2.X.shape}")
        # for i, t in enumerate(test_prot()):
        #     print(f"test_prot#{i} shape: {t.X.shape}")

        V_posteriors = model.predict_proba(V.X, V.attention_mask, verbose=True)
        print("V_posteriors")
        V1_posteriors = model.predict_proba(V1.X, V1.attention_mask, verbose=True)
        print("V1_posteriors")
        V2_prot_posteriors = []
        for sample in tqdm(V2_prot(), total=V2_prot.total()):
            V2_prot_posteriors.append(model.predict_proba(sample.X, sample.attention_mask))
        print("V2_prot_posteriors")

        sld_requa_params = {
            "add_X": [False],
            "add_posteriors": [True, False],
            "add_y_hat": [True, False],
            "add_maxconf": [True, False],
            "add_negentropy": [True, False],
            "add_maxinfsoft": [True, False],
        }

        sld_opt_params = sld_requa_params | {
            "q_class__classifier__C": np.logspace(-3, 3, 7),
            "q_class__classifier__class_weight": [None, "balanced"],
            "q_class__recalib": [None, "bcts"],
        }

        if BASE:
            quacc_n2 = QuAcc1xN2(
                vanilla_acc,
                sld(),
                add_X=False,
                add_y_hat=True,
                add_maxinfsoft=True,
            ).fit(V, V_posteriors)
            print("quacc_n2 fit")
            quacc_nn = QuAccNxN(
                vanilla_acc,
                sld(),
                add_X=False,
                add_y_hat=True,
                add_maxinfsoft=True,
            ).fit(V, V_posteriors)
            print("quacc_nn fit")
        if OPT_NN:
            quacc_nn_opt = GridSearchCAP(
                QuAccNxN(vanilla_acc, sld()), sld_opt_params, V2_prot, V2_prot_posteriors, vanilla_acc, refit=False
            ).fit(V1, V1_posteriors)
            print("quacc_nn_opt fit")
        if OPT_N2:
            quacc_n2_opt = GridSearchCAP(
                QuAcc1xN2(vanilla_acc, sld()), sld_opt_params, V2_prot, V2_prot_posteriors, vanilla_acc, refit=False
            ).fit(V1, V1_posteriors)
            print("quacc_n2_opt fit")
        if REQUA:
            requa = ReQua(
                vanilla_acc,
                KRR(),
                [QuAcc1xN2(vanilla_acc, sld()), QuAccNxN(vanilla_acc, sld()), QuAcc1xNp1(vanilla_acc, sld())],
                sld_requa_params,
                V2_prot,
                V2_prot_posteriors,
                n_jobs=0,
            ).fit(V1, V1_posteriors)
            print("requa fit")
        doc = DoC(vanilla_acc, V2_prot, V2_prot_posteriors).fit(V1, V1_posteriors)
        print("doc fit")

        test_y_hat, test_y = [], []
        if BASE:
            quacc_n2_accs = []
            quacc_nn_accs = []
        if OPT_NN:
            quacc_nn_opt_accs = []
        if OPT_N2:
            quacc_n2_opt_accs = []
        if REQUA:
            requa_accs = []
        doc_accs = []
        true_accs = []
        for i, U_i in enumerate(tqdm(test_prot(), total=test_prot.total())):
            P = model.predict_proba(U_i.X, U_i.attention_mask)
            y_hat = np.argmax(P, axis=-1)
            test_y_hat.append(y_hat)
            test_y.append(U_i.y)
            if BASE:
                quacc_nn_accs.append(quacc_nn.predict(U_i.X, P))
                print(f"quacc_nn prediction #{i}")
                quacc_n2_accs.append(quacc_n2.predict(U_i.X, P))
                print(f"quacc_n2 prediction #{i}")
            if OPT_NN:
                quacc_nn_opt_accs.append(quacc_nn_opt.predict(U_i.X, P))
                print(f"quacc_nn_opt prediction #{i}")
            if OPT_N2:
                quacc_n2_opt_accs.append(quacc_n2_opt.predict(U_i.X, P))
                print(f"quacc_n2_opt prediction #{i}")
            if REQUA:
                requa_accs.append(requa.predict(U_i.X, P))
                print(f"requa prediction #{i}")
            doc_accs.append(doc.predict(U_i.X, P))
            print(f"doc prediction #{i}")
            true_accs.append(vanilla_acc(y_hat, U_i.y))

        if BASE:
            quacc_n2_accs = np.asarray(quacc_n2_accs)
            quacc_nn_accs = np.asarray(quacc_nn_accs)
            quacc_n2_mean = np.mean(np.abs(quacc_n2_accs - true_accs))
            quacc_nn_mean = np.mean(np.abs(quacc_nn_accs - true_accs))
            results["quacc_nn"].append(quacc_nn_mean)
            results["quacc_n2"].append(quacc_n2_mean)
        if OPT_NN:
            quacc_nn_opt_accs = np.asarray(quacc_nn_opt_accs)
            quacc_nn_opt_mean = np.mean(np.abs(quacc_nn_opt_accs - true_accs))
            results["quacc_nn_opt"].append(quacc_nn_opt_mean)
        if OPT_N2:
            quacc_n2_opt_accs = np.asarray(quacc_n2_opt_accs)
            quacc_n2_opt_mean = np.mean(np.abs(quacc_n2_opt_accs - true_accs))
            results["quacc_n2_opt"].append(quacc_n2_opt_mean)
        if REQUA:
            requa_accs = np.asarray(requa_accs)
            requa_mean = np.mean(np.abs(requa_accs - true_accs))
            results["requa"].append(requa_mean)

        doc_accs = np.asarray(doc_accs)
        doc_mean = np.mean(np.abs(doc_accs - true_accs))
        results["doc"].append(doc_mean)

    df = pd.DataFrame(np.vstack(list(results.values())), columns=list(dataset_map.keys()), index=list(results.keys()))
    return df


if __name__ == "__main__":
    res = main()
    print(res)
