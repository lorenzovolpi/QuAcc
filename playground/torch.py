import os
import pdb
from typing import Callable, List

import datasets
import numpy as np
import quapy as qp
import torch
import torch.nn as nn
import torch.utils.data
from datasets import load_dataset
from quapy.data.base import LabelledCollection
from quapy.method.aggregative import SLD
from quapy.protocol import UPP
from scipy.sparse import issparse
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
from quacc.models.cont_table import QuAcc1xN2, QuAccNxN
from quacc.models.direct import DoC
from quacc.models.model_selection import GridSearchCAP
from quacc.models.utils import get_posteriors_from_h

NUM_SAMPLES = 100
SAMPLE_SIZE = 500
RANDOM_STATE = 42

qp.environ["_R_SEED"] = RANDOM_STATE
qp.environ["SAMPLE_SIZE"] = SAMPLE_SIZE


def softmax(logits: torch.Tensor) -> np.ndarray:
    return nn.functional.softmax(logits, dim=-1).numpy()


class LargeModel:
    def fit(self, train: LabelledCollection, dataset_name: str): ...

    def predict_proba(self, test) -> np.ndarray: ...

    def get_model_path(self, dataset_name):
        return os.path.join(qc.env["OUT_DIR"], "models", f"{self.name}_on_{dataset_name}")

    def predict_from_proba(self, proba: np.ndarray) -> np.ndarray:
        return np.argmax(proba, axis=-1)

    def predict(self, test) -> np.ndarray:
        return self.predict_from_proba(self.predict_proba(test))


class TorchLC(torch.utils.data.Dataset):
    def __init__(self, X, y, data_mapping, has_labels=True):
        self.X = X
        self.y = y
        self.data_mapping = data_mapping
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

        return {self.data_mapping[k]: v for k, v in res.items()}

    @classmethod
    def from_lc(cls, data, data_mapping):
        return TorchLC(*data.Xy, data_mapping)

    @classmethod
    def from_X(cls, data, data_mapping):
        return TorchLC(data, None, data_mapping, has_labels=False)


class DistilBert(LargeModel):
    data_mapping = {
        "X": "input_ids",
        "y": "labels",
    }

    def __init__(self, learning_rate=1e-5, batch_size=64, epochs=3, seed=RANDOM_STATE):
        self.name = "distilbert-base-uncased"
        self.tokenizer_name = self.name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = epochs
        self.seed = seed
        self.epoch = 0

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
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

    def fit(self, train: LabelledCollection, dataset_name: str):
        self.classes_ = train.classes_
        train_dl = DataLoader(
            TorchLC.from_lc(train, self.data_mapping),
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
                progress_bar.update(1)

        self.checkpoint(dataset_name)

        # TODO: unload model from CUDA

        return self

    def predict_proba(self, test) -> np.ndarray:
        # TODO: load model to cuda
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        test_dl = DataLoader(
            TorchLC.from_X(test, self.data_mapping),
            batch_size=self.batch_size,
        )
        self.model.to(device)

        self.model.eval()
        y_probs = []
        progress_bar = tqdm(range(len(test_dl)))
        with torch.no_grad():
            for batch in test_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                y_probs.append(softmax(outputs.logits.cpu()))
                progress_bar.update(1)

        # TODO: unload model from CUDA
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
) -> LabelledCollection:
    if remove_columns is not None:
        dataset = dataset.remove_columns(remove_columns)
    ds = next(iter(DataLoader(dataset, collate_fn=collator, batch_size=len(dataset))))
    return LabelledCollection(instances=ds["input_ids"], labels=ds["labels"])


def get_embeddings(model: DistilBert, data: LabelledCollection) -> LabelledCollection:
    device = torch.device("cuda")
    model.model.to(device)
    model.model.eval()
    data_dl = DataLoader(TorchLC.from_X(data.X, model.data_mapping), batch_size=model.batch_size)
    X_emb = []
    for batch in data_dl:
        input_ids = batch["input_ids"].to(device)
        batch_emb = model.model.get_input_embeddings()(input_ids)
        batch_emb = torch.flatten(batch_emb, start_dim=1)
        batch_emb = batch_emb.cpu().detach().numpy()
        X_emb.append(batch_emb)

    return LabelledCollection(np.vstack(X_emb), data.y, classes=data.classes_)


def mlp():
    MLP((100, 15), activation="logistic", solver="adam")


def lr():
    return LogisticRegression()


if __name__ == "__main__":
    model = DistilBert()
    dataset_name = "imdb"
    # dataset_name = "rotten_tomatoes"
    # dataset_name = "amazon_polarity"

    text_columns = ["text"]
    # text_columns = ["title", "content"]

    dataset = load_dataset(dataset_name)

    train_vec = preprocess_data(dataset, "train", model.tokenizer, text_columns, length=25000)
    test_vec = preprocess_data(dataset, "test", model.tokenizer, text_columns)

    train = from_hf_dataset(train_vec, model.data_collator, remove_columns=text_columns)
    U = from_hf_dataset(test_vec, model.data_collator, remove_columns=text_columns)
    L, V = train.split_stratified(train_prop=0.5, random_state=RANDOM_STATE)

    model.fit(L, dataset_name)

    test_prot = UPP(U, sample_size=SAMPLE_SIZE, repeats=NUM_SAMPLES, return_type="labelled_collection")

    V1, V2_prot = split_validation(V)

    V_posteriors = get_posteriors_from_h(model, V.X)
    print("V_posteriors")
    V1_posteriors = get_posteriors_from_h(model, V1.X)
    print("V1_posteriors")

    print(type(V1_posteriors))

    pdb.set_trace()

    sld_params = {
        "q_class__classifier__C": np.logspace(-3, 3, 7),
        "q_class__classifier__class_weight": [None, "balanced"],
        "add_X": [False],
        "add_posteriors": [True, False],
        "add_y_hat": [True, False],
        "add_maxconf": [True, False],
        "add_negentropy": [True, False],
        "add_maxinfsoft": [True, False],
        "q_class__recalib": [None, "bcts"],
    }

    quacc_n2 = QuAcc1xN2(
        model,
        vanilla_acc,
        SLD(lr()),
        add_X=False,
        add_y_hat=True,
        add_maxinfsoft=True,
    ).fit(V, posteriors=V_posteriors)
    print("quacc_n2 fit")
    quacc_nn = QuAccNxN(
        model,
        vanilla_acc,
        SLD(lr()),
        add_X=False,
        add_y_hat=True,
        add_maxinfsoft=True,
    ).fit(V, posteriors=V_posteriors)
    print("quacc_nn fit")
    quacc_nn_opt = GridSearchCAP(
        QuAccNxN(model, vanilla_acc, SLD(lr())), sld_params, V2_prot, vanilla_acc, refit=False
    ).fit(V1, posteriors=V1_posteriors)
    doc = DoC(model, vanilla_acc, sample_size=SAMPLE_SIZE, num_samples=NUM_SAMPLES).fit(V)
    print("doc fit")

    test_y_hat, test_y = [], []
    quacc_n2_accs, quacc_nn_accs, quacc_nn_opt_accs, doc_accs = [], [], [], []
    true_accs = []
    for U_i in test_prot():
        P = get_posteriors_from_h(model, U_i.X)
        y_hat = np.argmax(P, axis=-1)
        test_y_hat.append(y_hat)
        test_y.append(U_i.y)
        quacc_n2_accs.append(quacc_n2.predict(U_i.X, posteriors=P))
        quacc_nn_accs.append(quacc_nn.predict(U_i.X, posteriors=P))
        quacc_nn_opt_accs.append(quacc_nn_opt.predict(U_i.X, posteriors=P))
        doc_accs.append(doc.predict(U_i.X, posteriors=P))
        true_accs.append(vanilla_acc(y_hat, U_i.y))

    quacc_n2_accs = np.asarray(quacc_n2_accs)
    quacc_nn_accs = np.asarray(quacc_nn_accs)
    quacc_nn_opt_accs = np.asarray(quacc_nn_opt_accs)
    doc_accs = np.asarray(doc_accs)

    print(f"quacc_n2:\t{np.mean(np.abs(quacc_n2_accs - true_accs))}")
    print(f"quacc_nn:\t{np.mean(np.abs(quacc_nn_accs - true_accs))}")
    print(f"quacc_nn_opt:\t{np.mean(np.abs(quacc_nn_opt_accs - true_accs))}")
    print(f"doc:\t{np.mean(np.abs(doc_accs - true_accs))}")
