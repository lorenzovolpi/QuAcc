import os

import numpy as np
import torch
import torch.nn as nn
import datasets
from datasets import load_dataset
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
        self.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)

        return self

    def predict_proba(self, test: datasets.Dataset, get_labels=False) -> np.ndarray:
        y_logits, y, _ = self.trainer.predict(test)
        y_probs = softmax(y_logits)
        if get_labels:
            return y_probs, y
        else:
            return y_probs


def from_dataset(dataset, name, tokenizer, length: int | None = None, seed=42):
    def tokenize(datapoint):
        return tokenizer(datapoint["text"], truncation=True)

    d = dataset[name].shuffle(seed=seed)
    if length is not None:
        d = d.select(list(range(length)))
    d = d.map(tokenize, batched=True)
    return d


if __name__ == "__main__":
    model = DistilBert(resume_from_checkpoint=True)
    dataset_name = "imdb"

    dataset = load_dataset(dataset_name)

    train = from_dataset(dataset, "train", model.tokenizer)
    test = from_dataset(dataset, "test", model.tokenizer)

    model.fit(train, dataset_name)

    y_prob, y = model.predict_proba(test, get_labels=True)
    y_pred = model.predict_from_proba(y_prob)
    va = vanilla_acc(y, y_pred)

    print(
        f"{y_prob=}\n",
        f"y:\t{y}\n",
        f"y_pred:\t{y_pred}\n",
        f"{va=}",
    )
