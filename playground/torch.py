from transformers import AutoTokenizer
import json
import transformers
import torch
from datasets import load_dataset


if __name__ == "__main__":
    imdb = load_dataset("imdb")
    small_train = imdb["train"].shuffle(seed=42).select(list(range(3000)))
    small_test = imdb["test"].shuffle(seed=42).select(list(range(3000)))

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(datapoint):
        return tokenizer(datapoint["text"], truncation=True)

    train_pt = small_train.map(tokenize, batched=True).with_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    test_pt = small_test.map(tokenize, batched=True).with_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )

    print(train_pt[:10])

    # https://huggingface.co/docs/transformers/training
