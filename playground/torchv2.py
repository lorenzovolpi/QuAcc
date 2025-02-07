import quapy as qp
from datasets import load_dataset
from transformers import AutoTokenizer

qp.environ["_R_SEED"] = 0


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(datapoint):
        return tokenizer(datapoint["sentence"], padding="max_length", truncation=True)

    dataset = load_dataset("glue", "cola")
    tokenized_dataset = dataset.map(tokenize, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns("sentence")
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("pytorch")
    print(tokenized_dataset)
    print(type(tokenized_dataset["train"]))
    print(tokenized_dataset["train"].select([135, 189, 256]))

    # test set
    # d = dataset["test"].shuffle(seed=qp.environ["_R_SEED"])
    # if length is not None:
    #     d = d.select(np.arange(length))
    # d = d.map(tokenize, batched=True)
