import os
from typing import List

import datasets
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from quapy.data.base import LabelledCollection
from quapy.protocol import UPP
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import quacc as qc
from quacc.error import vanilla_acc

RESUME_CHECKPOINT = True


def softmax(logits: np.ndarray) -> np.ndarray:
    sm = nn.Softmax(dim=-1)
    return sm(torch.tensor(logits)).numpy()


class LargeModel:
    def fit(self, train: datasets.Dataset, dataset_name: str): ...

    def predict_proba(self, test: datasets.Dataset) -> np.ndarray: ...

    def get_model_path(self, dataset_name):
        return os.path.join(qc.env["OUT_DIR"], "models", f"{self.name}_on_{dataset_name}")

    def predict_from_proba(self, proba: np.ndarray) -> np.ndarray:
        return np.argmax(proba, axis=-1)

    def predict(self, test: datasets.Dataset) -> np.ndarray:
        return self.predict_from_proba(self.predict_proba(test))


class DistilBert(LargeModel):
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

    def fit(self, train: datasets.Dataset, dataset_name: str):
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
            train_dataset=train,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        self.trainer.train(
            resume_from_checkpoint=self._check_checkpoint(dataset_name),
        )

        return self

    def predict_proba(self, test: datasets.Dataset, get_labels=False) -> np.ndarray:
        y_logits, y, _ = self.trainer.predict(test)
        y_probs = softmax(y_logits)
        if get_labels:
            return y_probs, y
        else:
            return y_probs


def preprocess_data(dataset, name, tokenizer, length: int | None = None, seed=42) -> datasets.Dataset:
    def tokenize(datapoint):
        return tokenizer(datapoint["text"], truncation=True)

    d = dataset[name].shuffle(seed=seed)
    if length is not None:
        d = d.select(list(range(length)))
    d = d.map(tokenize, batched=True)
    return d


def lc_from_dataset(
    dataset: datasets.Dataset, collator, remove_columns: str | List[str] | None = None
) -> LabelledCollection:
    if remove_columns is not None:
        dataset = dataset.remove_columns(remove_columns)
    dl = DataLoader(dataset, collate_fn=collator, batch_size=len(dataset))
    ds = next(dl)
    return LabelledCollection(instances=ds["input_ids"], labels=ds["label"])


if __name__ == "__main__":
    model = DistilBert(resume_from_checkpoint=True)
    dataset_name = "imdb"

    dataset = load_dataset(dataset_name)

    train = preprocess_data(dataset, "train", model.tokenizer)
    test = preprocess_data(dataset, "test", model.tokenizer)

    train_lc = lc_from_dataset(train, model.data_collator, remove_columns="text")
    test_lc = lc_from_dataset(test, model.data_collator, remove_columns="text")

    L_idx, V_idx = None, None

    model.fit(train, dataset_name)

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
    # test_prot = UPP(test_lc, sample_size=1000, repeats=10, return_type="index")
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
