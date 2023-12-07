import numpy as np
import pytest
import scipy.sparse as sp

from quacc.data import (
    ExtendedCollection,
    ExtendedData,
    ExtendedLabels,
    ExtendedPrev,
    ExtensionPolicy,
)


@pytest.mark.ext
@pytest.mark.extpol
class TestExtendedPolicy:
    @pytest.mark.parametrize(
        "extpol,nbcl,result",
        [
            (ExtensionPolicy(), 2, np.array([0, 1, 2, 3])),
            (ExtensionPolicy(collapse_false=True), 2, np.array([0, 1, 2])),
            (ExtensionPolicy(), 3, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])),
            (ExtensionPolicy(collapse_false=True), 3, np.array([0, 1, 2, 3])),
        ],
    )
    def test_qclasses(self, extpol, nbcl, result):
        assert (result == extpol.qclasses(nbcl)).all()

    @pytest.mark.parametrize(
        "extpol,nbcl,result",
        [
            (ExtensionPolicy(), 2, np.array([0, 1, 2, 3])),
            (ExtensionPolicy(collapse_false=True), 2, np.array([0, 1, 2, 3])),
            (ExtensionPolicy(), 3, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])),
            (
                ExtensionPolicy(collapse_false=True),
                3,
                np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
            ),
        ],
    )
    def test_eclasses(self, extpol, nbcl, result):
        assert (result == extpol.eclasses(nbcl)).all()

    @pytest.mark.parametrize(
        "extpol,nbcl,result",
        [
            (
                ExtensionPolicy(),
                2,
                (
                    np.array([0, 0, 1, 1]),
                    np.array([0, 1, 0, 1]),
                ),
            ),
            (
                ExtensionPolicy(collapse_false=True),
                2,
                (
                    np.array([0, 1, 0]),
                    np.array([0, 1, 1]),
                ),
            ),
            (
                ExtensionPolicy(),
                3,
                (
                    np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                    np.array([0, 1, 2, 0, 1, 2, 0, 1, 2]),
                ),
            ),
            (
                ExtensionPolicy(collapse_false=True),
                3,
                (
                    np.array([0, 1, 2, 0]),
                    np.array([0, 1, 2, 1]),
                ),
            ),
        ],
    )
    def test_matrix_idx(self, extpol, nbcl, result):
        _midx = extpol.matrix_idx(nbcl)
        assert len(_midx) == len(result)
        assert all((idx == r).all() for idx, r in zip(_midx, result))

    @pytest.mark.parametrize(
        "extpol,nbcl,true,pred,result",
        [
            (
                ExtensionPolicy(),
                2,
                np.array([1, 0, 1, 1, 0, 0]),
                np.array([1, 0, 0, 1, 1, 0]),
                np.array([3, 0, 2, 3, 1, 0]),
            ),
            (
                ExtensionPolicy(collapse_false=True),
                2,
                np.array([1, 0, 1, 1, 0, 0]),
                np.array([1, 0, 0, 1, 1, 0]),
                np.array([1, 0, 2, 1, 2, 0]),
            ),
            (
                ExtensionPolicy(),
                3,
                np.array([1, 2, 0, 1, 0, 2, 0, 1, 2]),
                np.array([1, 0, 0, 0, 1, 1, 2, 2, 2]),
                np.array([4, 6, 0, 3, 1, 7, 2, 5, 8]),
            ),
            (
                ExtensionPolicy(collapse_false=True),
                3,
                np.array([1, 2, 0, 1, 0, 2, 0, 1, 2]),
                np.array([1, 0, 0, 0, 1, 1, 2, 2, 2]),
                np.array([1, 3, 0, 3, 3, 3, 3, 3, 2]),
            ),
        ],
    )
    def test_ext_lbl(self, extpol, nbcl, true, pred, result):
        vfun = extpol.ext_lbl(nbcl)
        assert (vfun(true, pred) == result).all()


@pytest.mark.ext
@pytest.mark.extd
class TestExtendedData:
    @pytest.mark.parametrize(
        "pred_proba,result",
        [
            (
                np.array([[0.3, 0.7], [0.54, 0.46], [0.28, 0.72], [0.6, 0.4]]),
                [np.array([1, 3]), np.array([0, 2])],
            ),
            (
                np.array([[0.3, 0.7], [0.28, 0.72]]),
                [np.array([]), np.array([0, 1])],
            ),
            (
                np.array([[0.54, 0.46], [0.6, 0.4]]),
                [np.array([0, 1]), np.array([])],
            ),
            (
                np.array(
                    [
                        [0.25, 0.4, 0.35],
                        [0.24, 0.3, 0.46],
                        [0.61, 0.28, 0.11],
                        [0.4, 0.1, 0.5],
                    ]
                ),
                [np.array([2]), np.array([0]), np.array([1, 3])],
            ),
        ],
    )
    def test__split_index_by_pred(self, monkeypatch, pred_proba, result):
        def mockinit(self, pred_proba):
            self.pred_proba_ = pred_proba

        monkeypatch.setattr(ExtendedData, "__init__", mockinit)
        ed = ExtendedData(pred_proba)
        _split_index = ed._ExtendedData__split_index_by_pred()
        assert len(_split_index) == len(result)
        assert all((a == b).all() for (a, b) in zip(_split_index, result))


@pytest.mark.ext
@pytest.mark.extl
class TestExtendedLabels:
    @pytest.mark.parametrize(
        "true,pred,nbcl,extpol,result",
        [
            (
                np.array([1, 0, 0, 1, 1]),
                np.array([1, 1, 0, 0, 1]),
                2,
                ExtensionPolicy(),
                np.array([3, 1, 0, 2, 3]),
            ),
            (
                np.array([1, 0, 0, 1, 1]),
                np.array([1, 1, 0, 0, 1]),
                2,
                ExtensionPolicy(collapse_false=True),
                np.array([1, 2, 0, 2, 1]),
            ),
        ],
    )
    def test_y(self, true, pred, nbcl, extpol, result):
        el = ExtendedLabels(true, pred, nbcl, extpol)
        assert (el.y == result).all()


@pytest.mark.ext
@pytest.mark.extp
class TestExtendedPrev:
    @pytest.mark.parametrize(
        "flat,nbcl,extpol,q_classes,result",
        [
            (
                np.array([0.2, 0, 0.8, 0]),
                2,
                ExtensionPolicy(),
                [0, 1, 2, 3],
                np.array([0.2, 0, 0.8, 0]),
            ),
            (
                np.array([0.2, 0.8]),
                2,
                ExtensionPolicy(),
                [0, 3],
                np.array([0.2, 0, 0, 0.8]),
            ),
            (
                np.array([0.2, 0.8]),
                2,
                ExtensionPolicy(collapse_false=True),
                [0, 2],
                np.array([0.2, 0, 0.8]),
            ),
            (
                np.array([0.1, 0.1, 0.6, 0.2]),
                3,
                ExtensionPolicy(),
                [0, 1, 3, 5],
                np.array([0.1, 0.1, 0, 0.6, 0, 0.2, 0, 0, 0]),
            ),
            (
                np.array([0.1, 0.1, 0.6]),
                3,
                ExtensionPolicy(collapse_false=True),
                [0, 1, 2],
                np.array([0.1, 0.1, 0.6, 0]),
            ),
        ],
    )
    def test__check_q_classes(self, monkeypatch, flat, nbcl, extpol, q_classes, result):
        def mockinit(self, flat, nbcl, extpol):
            self.flat = flat
            self.nbcl = nbcl
            self.extpol = extpol

        monkeypatch.setattr(ExtendedPrev, "__init__", mockinit)
        ep = ExtendedPrev(flat, nbcl, extpol)
        ep._ExtendedPrev__check_q_classes(q_classes)
        assert (ep.flat == result).all()

    @pytest.mark.parametrize(
        "flat,nbcl,extpol,result",
        [
            (
                np.array([0.05, 0.1, 0.6, 0.25]),
                2,
                ExtensionPolicy(),
                np.array([[0.05, 0.1], [0.6, 0.25]]),
            ),
            (
                np.array([0.05, 0.1, 0.85]),
                2,
                ExtensionPolicy(collapse_false=True),
                np.array([[0.05, 0.85], [0, 0.1]]),
            ),
            (
                np.array([0.05, 0.1, 0.2, 0.15, 0.04, 0.06, 0.15, 0.14, 0.1]),
                3,
                ExtensionPolicy(),
                np.array([[0.05, 0.1, 0.2], [0.15, 0.04, 0.06], [0.15, 0.14, 0.1]]),
            ),
            (
                np.array([0.05, 0.2, 0.65, 0.1]),
                3,
                ExtensionPolicy(collapse_false=True),
                np.array([[0.05, 0.1, 0], [0, 0.2, 0], [0, 0, 0.65]]),
            ),
        ],
    )
    def test__build_matrix(self, monkeypatch, flat, nbcl, extpol, result):
        def mockinit(self, flat, nbcl, extpol):
            self.flat = flat
            self.nbcl = nbcl
            self.extpol = extpol

        monkeypatch.setattr(ExtendedPrev, "__init__", mockinit)
        ep = ExtendedPrev(flat, nbcl, extpol)
        _matrix = ep._ExtendedPrev__build_matrix()
        assert _matrix.shape == result.shape
        assert (_matrix == result).all()
