import os
from contextlib import redirect_stderr

import numpy as np
import pytest

from quacc.dataset import Dataset


@pytest.mark.dataset
class TestDataset:
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "name,target,prevalence",
        [
            ("spambase", None, [0.5, 0.5]),
            ("imdb", None, [0.5, 0.5]),
            ("rcv1", "CCAT", [0.5, 0.5]),
            ("cifar10", "dog", [0.5, 0.5]),
            ("twitter_gasp", None, [0.33, 0.33, 0.33]),
        ],
    )
    def test__resample_all_train(self, name, target, prevalence, monkeypatch):
        def mockinit(self):
            self._name = name
            self._target = target
            self.all_train, self.test = self.alltrain_test(self._name, self._target)

        monkeypatch.setattr(Dataset, "__init__", mockinit)
        with open(os.devnull, "w") as dn:
            with redirect_stderr(dn):
                d = Dataset()
                d._Dataset__resample_all_train()
                assert (
                    np.around(d.all_train.prevalence(), decimals=2).tolist()
                    == prevalence
                )

    @pytest.mark.parametrize(
        "ncl, prevs,result",
        [
            (2, None, None),
            (2, [], None),
            (2, [[0.2, 0.1], [0.3, 0.2]], None),
            (2, [[0.2, 0.8], [0.3, 0.7]], [[0.2, 0.8], [0.3, 0.7]]),
            (2, [1.0, 2.0, 3.0], None),
            (2, [1, 2, 3], None),
            (2, [[1, 2], [2, 3], [3, 4]], None),
            (2, ["abc", "def"], None),
            (3, [[0.2, 0.3], [0.4, 0.1], [0.5, 0.2]], None),
            (3, [[0.2, 0.3, 0.2], [0.4, 0.1], [0.5, 0.6]], None),
            (2, [[0.2, 0.3, 0.1], [0.1, 0.5, 0.3]], None),
            (3, [[0.2, 0.3, 0.1], [0.1, 0.5, 0.3]], None),
            (3, [[0.2, 0.8], [0.1, 0.5]], None),
            (2, [[0.2, 0.9], [0.1, 0.5]], None),
            (2, 10, None),
            (2, [[0.2, 0.8], [0.5, 0.5]], [[0.2, 0.8], [0.5, 0.5]]),
            (3, [[0.2, 0.6], [0.3, 0.5]], None),
        ],
    )
    def test__check_prevs(self, ncl, prevs, result, monkeypatch):
        class MockLabelledCollection:
            def __init__(self):
                self.n_classes = ncl

        def mockinit(self):
            self.all_train = MockLabelledCollection()
            self.prevs = None

        monkeypatch.setattr(Dataset, "__init__", mockinit)
        d = Dataset()
        d._Dataset__check_prevs(prevs)
        _prevs = d.prevs if d.prevs is None else d.prevs.tolist()
        assert _prevs == result

    # fmt: off


    @pytest.mark.parametrize(
        "ncl,nprevs,built,result",
        [
            (2, 3, None, [[0.25, 0.75], [0.5, 0.5], [0.75, 0.25]]),
            (2, 3, np.array([[0.8, 0.2], [0.6, 0.4], [0.4, 0.6]]), [[0.8, 0.2], [0.6, 0.4], [0.4, 0.6]]),
            (2, 3, np.array([[0.75, 0.25], [0.5, 0.5], [0.25, 0.75]]), [[0.75, 0.25], [0.5, 0.5], [0.25, 0.75]]),
            (3, 3, None, [[0.25, 0.25, 0.5], [0.25, 0.5, 0.25], [0.5, 0.25, 0.25]]),
            (
                3, 4, None,
                [[0.2, 0.2, 0.6], [0.2, 0.4, 0.4], [0.2, 0.6, 0.2], [0.4, 0.2, 0.4], [0.4, 0.4, 0.2], [0.6, 0.2, 0.2]],
            ),
        ],
    )
    def test__build_prevs(self, ncl, nprevs, built, result, monkeypatch):
        class MockLabelledCollection:
            def __init__(self):
                self.n_classes = ncl

        def mockinit(self):
            self.all_train = MockLabelledCollection()
            self.prevs = built
            self._n_prevs = nprevs

        monkeypatch.setattr(Dataset, "__init__", mockinit)
        d = Dataset()
        _prevs = d._Dataset__build_prevs().tolist() 
        assert  _prevs == result

    # fmt: on

    @pytest.mark.parametrize(
        "ncl,prevs,atsize",
        [
            (2, np.array([[0.2, 0.8], [0.9, 0.1]]), 55),
            (3, np.array([[0.2, 0.7, 0.1], [0.9, 0.05, 0.05]]), 37),
        ],
    )
    def test_get(self, ncl, prevs, atsize, monkeypatch):
        class MockLabelledCollection:
            def __init__(self):
                self.n_classes = ncl

            def __len__(self):
                return 100

        def mockinit(self):
            self.prevs = prevs
            self.all_train = MockLabelledCollection()

        def mock_build_sample(self, p, at_size):
            return at_size

        monkeypatch.setattr(Dataset, "__init__", mockinit)
        monkeypatch.setattr(Dataset, "_Dataset__build_sample", mock_build_sample)
        d = Dataset()
        _get = d.get()
        assert all(s == atsize for s in _get)
