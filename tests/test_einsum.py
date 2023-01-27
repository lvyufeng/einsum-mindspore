import mindspore
import numpy as np
from src.einsum import einsum

np.random.seed(102)

def test_group_1():
    # -------------------
    # Group 1
    # -------------------
    shapes = [
        [1, 5, 2, 2, 3, 4],
        [5, 2, 3, 4],
        [2, 1, 2],
        [1, 5, 2, 3, 4]
    ]
    x, y, z, t = [np.random.rand(*s) for s in shapes]
    tx, ty, tz, tt = [mindspore.Tensor(_) for _ in (x, y, z, t)]

    equations = [
        'abcdef, bcef, cad'
    ]
    for e in equations:
        np_out = np.einsum(e, x, y, z)
        ms_out = einsum(e, tx, ty, tz).asnumpy()
        assert np.allclose(np_out, ms_out)


def test_group_2():
    # -------------------
    # Group 2
    # -------------------
    shapes = [
        [5, 1, 10000],
        [100, 10000]
    ]
    x, y = [np.random.randn(*s) for s in shapes]
    tx, ty = [mindspore.Tensor(_) for _ in (x, y)]

    equations = [
        'ijk->j'
    ]

    for eqn in equations:
        np_out = np.einsum(eqn, x)
        ms_out = einsum(eqn, tx).asnumpy()
        assert np.allclose(np_out, ms_out)

    equations = [               \
        'ijk, jk',              \
        # '...k, ...k->...k',     \
        # 'ij..., j...',          \
        # 'ij..., j...->...'      \
    ]

    for eqn in equations:
        np_out = np.einsum(eqn, x, y)
        ms_out = einsum(eqn, tx, ty).asnumpy()
        assert np.allclose(np_out, ms_out)


def test_group_3():
    # -------------------
    # Group 3
    # -------------------
    shapes = [
        [4],
        [5]
    ]
    x, y = [np.random.randn(*s) for s in shapes]
    tx, ty = [mindspore.Tensor(_) for _ in (x, y)]

    equations = [
        'i,i->'
    ]
    for eqn in equations:
        np_out = np.einsum(eqn, x, x)
        ms_out = einsum(eqn, tx, tx).asnumpy()
        assert np.allclose(np_out, ms_out)

    equations =[
        'i,j->',
        'i,j->ij'
    ]
    for eqn in equations:
        np_out = np.einsum(eqn, x, y)
        ms_out = einsum(eqn, tx, ty).asnumpy()
        assert np.allclose(np_out, ms_out)

def test_group_4():
    # -------------------
    # Group 4
    # -------------------
    shapes = [
        [10, 1, 4, 256],
        [256, 10, 1]
    ]
    x, y = [np.random.randn(*s) for s in shapes]
    tx, ty = [mindspore.Tensor(_) for _ in (x, y)]

    equations = [
        'abcd,dfg->d'
    ]
    for eqn in equations:
        np_out = np.einsum(eqn, x, y)
        ms_out = einsum(eqn, tx, ty).asnumpy()
        assert np.allclose(np_out, ms_out)


def test_group_5():
    # -------------------
    # Group 5
    # -------------------
    shapes = [
        [10000, 100, 10]
    ]
    x = np.random.randn(*shapes[0])
    tx = mindspore.Tensor(x)

    equations = [
        'ijk->'
    ]

    for eqn in equations:
        np_out = np.einsum(eqn, x)
        ms_out = einsum(eqn, tx).asnumpy()
        assert np.allclose(np_out, ms_out)

def test_einsum_jit():
    @mindspore.jit
    def einsum_test(x):
        return einsum('ijk->', x)

    shapes = [
        [10000, 100, 10]
    ]
    x = np.random.randn(*shapes[0])
    tx = mindspore.Tensor(x)
    ms_out = einsum_test(tx)
