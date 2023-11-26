from contextlib import contextmanager

import yaml


class environ:
    _default_env = {
        "DATASET_NAME": None,
        "DATASET_TARGET": None,
        "METRICS": [],
        "COMP_ESTIMATORS": [],
        "DATASET_N_PREVS": 9,
        "DATASET_PREVS": None,
        "OUT_DIR_NAME": "output",
        "OUT_DIR": None,
        "PLOT_DIR_NAME": "plot",
        "PLOT_OUT_DIR": None,
        "DATASET_DIR_UPDATE": False,
        "PROTOCOL_N_PREVS": 21,
        "PROTOCOL_REPEATS": 100,
        "SAMPLE_SIZE": 1000,
        # "PLOT_ESTIMATORS": [],
        "PLOT_STDEV": False,
        "_R_SEED": 0,
    }
    _keys = list(_default_env.keys())

    def __init__(self):
        self.__load_file()

    def __load_file(self):
        _state = environ._default_env.copy()

        with open("conf.yaml", "r") as f:
            confs = yaml.safe_load(f)["exec"]

        _state = _state | confs["global"]
        self.__setdict(_state)
        self._confs = confs["confs"]

    def __setdict(self, d: dict):
        for k, v in d.items():
            super().__setattr__(k, v)

    def __getdict(self) -> dict:
        return {k: self.__getattribute__(k) for k in environ._keys}

    @property
    def confs(self):
        return self._confs.copy()

    @contextmanager
    def load(self, conf):
        __current = self.__getdict()
        if conf is not None:
            if isinstance(conf, dict):
                self.__setdict(conf)
            elif isinstance(conf, environ):
                self.__setdict(conf.__getdict())
        try:
            yield
        finally:
            self.__setdict(__current)

    def load_confs(self):
        for c in self.confs:
            with self.load(c):
                yield c


env = environ()
