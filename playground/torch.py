import os
from typing import Callable, List

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from datasets import load_dataset
from datasets.load import DataFilesPatternsList
from quapy.data.base import LabelledCollection
from quapy.method.aggregative import SLD
from quapy.protocol import UPP
from scipy.sparse import issparse
from sklearn.linear_model import LogisticRegression
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    get_scheduler,
)

import quacc as qc
from quacc.error import vanilla_acc
from quacc.models.cont_table import QuAcc1xN2
from quacc.models.direct import DoC

RESUME_CHECKPOINT = True


def softmax(logits: torch.Tensor) -> np.ndarray:
    sm = nn.Softmax(dim=-1)
    return sm(logits).numpy()


class HFDatasetAdapter(LabelledCollection):
    def __init__(self, instances, labels, hf: datasets.Dataset, classes=None):
        super().__init__(instances, labels, classes=classes)
        self.hf = hf

    @classmethod
    def from_hf_dataset(
        cls, dataset: datasets.Dataset, collator: Callable, remove_columns: str | List[str] | None = None
    ) -> "HFDatasetAdapter":
        if remove_columns is not None:
            dataset = dataset.remove_columns(remove_columns)
        ds = next(iter(DataLoader(dataset, collate_fn=collator, batch_size=len(dataset))))
        lc = HFDatasetAdapter(instances=ds["input_ids"], labels=ds["labels"], hf=dataset)

        return lc

    def sampling_from_index(self, index):
        instances = self.instances[index]
        labels = self.labels[index]
        hf = self.hf.select(index)
        return HFDatasetAdapter(instances, labels, hf, classes=self.classes_)

    def split_stratified(self, train_prop=0.6, random_state=None):
        train_idx, test_idx = super().split_index_stratified(train_prop, random_state)

        train = self.sampling_from_index(train_idx)
        test = self.sampling_from_index(test_idx)

        return train, test


class LargeModel:
    def fit(self, train: LabelledCollection, dataset_name: str): ...

    def predict_proba(self, test) -> np.ndarray: ...

    def get_model_path(self, dataset_name):
        return os.path.join(qc.env["OUT_DIR"], "models", f"{self.name}_on_{dataset_name}")

    def predict_from_proba(self, proba: np.ndarray) -> np.ndarray:
        return np.argmax(proba, axis=-1)

    def predict(self, test) -> np.ndarray:
        return self.predict_from_proba(self.predict_proba(test))


class DistilBert2(LargeModel):
    def __init__(self, learning_rate=1e-5, batch_size=64, epochs=3, seed=42, resume_from_checkpoint=True):
        self.name = "distilbert-base-uncased"
        self.tokenizer_name = self.name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.resume_from_checkpoint = resume_from_checkpoint

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.name)

    def get_model_path(self, dataset_name):
        return os.path.join(qc.env["OUT_DIR"], "models", f"{self.name}_on_{dataset_name}")

    def _check_checkpoint(self, dataset_name):
        if self.resume_from_checkpoint:
            model_path = self.get_model_path(dataset_name)
            return os.path.exists(model_path) and len(os.listdir(model_path)) > 0

        return self.resume_from_checkpoint

    def fit(self, train: HFDatasetAdapter, dataset_name: str):
        self.training_args = TrainingArguments(
            output_dir=self.get_model_path(dataset_name),
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            save_strategy="steps",
            seed=self.seed,
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train.hf,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        self.trainer.train(
            resume_from_checkpoint=self._check_checkpoint(dataset_name),
        )

        return self

    def predict_proba(self, test: HFDatasetAdapter, get_labels=False) -> np.ndarray:
        y_logits, y, _ = self.trainer.predict(test.hf)
        y_probs = softmax(y_logits)
        if get_labels:
            return y_probs, y
        else:
            return y_probs


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

    def __init__(self, learning_rate=1e-5, batch_size=64, epochs=3, seed=42):
        self.name = "distilbert-base-uncased"
        self.tokenizer_name = self.name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = epochs
        self.seed = seed
        self.epoch = 0

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.name)

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

        return self

    def predict_proba(self, test) -> np.ndarray:
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

        return np.vstack(y_probs)


def preprocess_data(dataset, name, tokenizer, length: int | None = None, seed=42) -> datasets.Dataset:
    def tokenize(datapoint):
        return tokenizer(datapoint["text"], truncation=True)

    d = dataset[name].shuffle(seed=seed)
    if length is not None:
        d = d.select(np.arange(length))
    d = d.map(tokenize, batched=True)
    return d


if __name__ == "__main__":
    model = DistilBert()
    dataset_name = "imdb"

    dataset = load_dataset(dataset_name)

    train_vec = preprocess_data(dataset, "train", model.tokenizer)
    test_vec = preprocess_data(dataset, "test", model.tokenizer)

    train = HFDatasetAdapter.from_hf_dataset(train_vec, model.data_collator, remove_columns="text")
    U = HFDatasetAdapter.from_hf_dataset(test_vec, model.data_collator, remove_columns="text")
    L, V = train.split_stratified(train_prop=0.5, random_state=42)

    model.fit(L, dataset_name)

    test_prot = UPP(U, sample_size=1000, repeats=100, return_type="labelled_collection")

    quacc = QuAcc1xN2(model, vanilla_acc, SLD(LogisticRegression()), add_maxinfsoft=True).fit(V)
    print("quacc fit")
    doc = DoC(model, vanilla_acc, sample_size=1000).fit(V)
    print("doc fit")

    quacc_accs = [quacc.predict(U_i.X) for U_i in test_prot()]
    print(f"quacc accs:\t{np.mean(quacc_accs)}")
    doc_accs = [doc.predict(U_i.X) for U_i in test_prot()]
    print(f"doc accs:\t{np.mean(doc_accs)}")

    true_accs = [vanilla_acc(model.predict(U_i.X), U_i.y) for U_i in test_prot()]

    print(f"quacc:\t{np.mean(quacc_accs - true_accs)}")
    print(f"doc:\t{np.mean(doc_accs - true_accs)}")

    # change the protocol collator

    # print(test.with_format)
    # ams = []
    # for am in test["attention_mask"]:
    #     ams.append(len(am))
    # print(np.min(ams), np.max(ams))
    #
    # dl = DataLoader(test.remove_columns("text"), collate_fn=model.data_collator, batch_size=model.batch_size)
    # n_feats = [d["input_ids"].shape[1] for d in dl]
    # print(n_feats)

    # min_lens = []
    # max_lens = []
    # for i, id in enumerate(test["input_ids"]):
    #     if (i % model.batch_size) == 0:
    #         max_lens.append(0)
    #         min_lens.append(1000)
    #     idl = len(id)
    #     if idl > max_lens[-1]:
    #         max_lens[-1] = idl
    #     if idl < min_lens[-1]:
    #         min_lens[-1] = idl
    #
    # print(max_lens)
    # print(min_lens)

    # test_lc = LabelledCollection(instances=np.empty((len(test), 1)), labels=test["label"])
    # test_samples = [test.select(idx) for idx in test_prot()]
    # for sample in test_samples:
    #     y_prob, y = model.predict_proba(sample, get_labels=True)
    #     y_pred = model.predict_from_proba(y_prob)
    #     va = vanilla_acc(y, y_pred)
    #     print(
    #         # f"{y_prob=}\n",
    #         # f"y:\t{y}\n",
    #         # f"y_pred:\t{y_pred}\n",
    #         f"{va=}",
    #     )
