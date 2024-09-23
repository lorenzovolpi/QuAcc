from typing import Callable, List

import datasets
import numpy as np
import quapy as qp
from datasets import load_dataset
from torch.utils.data import DataLoader

from quacc.data.base import TorchLabelledCollection


def preprocess_data(dataset, name, tokenizer, columns, length: int | None = None) -> datasets.Dataset:
    def tokenize(datapoint):
        sentences = [datapoint[c] for c in columns]
        return tokenizer(*sentences, truncation=True)

    d = dataset[name].shuffle(seed=qp.environ["_R_SEED"])
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
    return TorchLabelledCollection(instances=ds["input_ids"], labels=ds["labels"], attention_mask=ds["attention_mask"])


def fetch_imdbHFDataset(tokenizer, data_collator, train_length=25000, with_name=True):
    dataset_name = "imdb"
    text_columns = ["text"]
    dataset = load_dataset(dataset_name)

    train_vec = preprocess_data(dataset, "train", tokenizer, text_columns, length=train_length)
    test_vec = preprocess_data(dataset, "test", tokenizer, text_columns)

    train = from_hf_dataset(train_vec, data_collator, remove_columns=text_columns)
    U = from_hf_dataset(test_vec, data_collator, remove_columns=text_columns)
    L, V = train.split_stratified(train_prop=0.5, random_state=qp.environ["_R_SEED"])

    if with_name:
        return dataset_name, (L, V, U)

    return L, V, U


def fetch_rottenTomatoesHFDataset(tokenizer, data_collator, train_length=8530, with_name=True):
    dataset_name = "rotten_tomatoes"
    text_columns = ["text"]
    dataset = load_dataset(dataset_name)

    train_vec = preprocess_data(dataset, "train", tokenizer, text_columns, length=train_length)
    test_vec = preprocess_data(dataset, "test", tokenizer, text_columns)

    train = from_hf_dataset(train_vec, data_collator, remove_columns=text_columns)
    U = from_hf_dataset(test_vec, data_collator, remove_columns=text_columns)
    L, V = train.split_stratified(train_prop=0.5, random_state=qp.environ["_R_SEED"])

    if with_name:
        return dataset_name, (L, V, U)

    return L, V, U


def fetch_amazonPolarityHFDataset(tokenizer, data_collator, train_length=25000, with_name=True):
    dataset_name = "amazon_polarity"
    text_columns = ["title", "content"]
    dataset = load_dataset(dataset_name)

    train_vec = preprocess_data(dataset, "train", tokenizer, text_columns, length=train_length)
    test_vec = preprocess_data(dataset, "test", tokenizer, text_columns)

    train = from_hf_dataset(train_vec, data_collator, remove_columns=text_columns)
    U = from_hf_dataset(test_vec, data_collator, remove_columns=text_columns)
    L, V = train.split_stratified(train_prop=0.5, random_state=qp.environ["_R_SEED"])

    if with_name:
        return dataset_name, (L, V, U)

    return L, V, U
