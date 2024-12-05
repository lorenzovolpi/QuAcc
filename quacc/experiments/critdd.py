from quapy.data.datasets import UCI_BINARY_DATASETS

PROBLEM = "binary"


def gen_acc_names():
    yield "vanilla_accuracy"


def get_dataset_names():
    if PROBLEM == "binary":
        _uci_skip = ["acute.a", "acute.b", "balance.2", "iris.1"]
        return [d for d in UCI_BINARY_DATASETS if d not in _uci_skip]


def gen_classifier_names():
    return ["LR"]


def get_method_names():
    return [
        "DoC",
        "LEAP(SLD)",
        "OCE(SLD)",
        "PHD(SLD)",
        "ReQua(SLD-KRR)",
        "ReQua(SLD-Ridge)",
    ]


def plotting():
    pass


if __name__ == "__main__":
    plotting()
