import numpy as np
import pytest

from JMIM.entropy import *
from JMIM.entropy import _invert_axes

@pytest.fixture(scope="module", params=list(range(6)))
def _random_dataset(request):
    np.random.seed(request.param)

    nfeatures = np.random.randint(3, 10)
    rows = np.random.randint(10_000, 50_000)
    Y_axes = tuple(range(1, np.random.randint(2, nfeatures+1)))

    data = np.random.randint(0, 10, size=(rows, nfeatures))
    label_data(data)

    pmf = generate_pmf(data)

    return [pmf, Y_axes]


def _conditional_entropy_2(joint_pmf, Y_axes=(-1,)):
    """For testing purposes"""

    ndim = joint_pmf.ndim

    #Ensure axes are in the interval [0, ndim) (Also allows for negative indices).
    Y_axes = np.mod(Y_axes, ndim)
    X_axes = _invert_axes(Y_axes, ndim)

    # expand_dims for correct broadcasting with joint_pmf
    Y_pmf = np.expand_dims(np.sum(joint_pmf, axis=X_axes), X_axes)
    # (May throw divide by zero warning)
    return -np.sum(joint_pmf[joint_pmf > 0] * np.log2((joint_pmf / Y_pmf)[joint_pmf > 0]))


def test_entropy_compare(_random_dataset):
    e1 = conditional_entropy(_random_dataset[0], _random_dataset[1])
    e2 = _conditional_entropy_2(_random_dataset[0], _random_dataset[1])

    assert np.abs(e1 - e2) < 1e-8


def _MI_2(joint_pmf, Y_axes=(-1,)):
    ndim = joint_pmf.ndim

    #Ensure axes are in the interval [0, ndim) (Also allows for negative indices).
    Y_axes = tuple(np.mod(Y_axes, ndim))
    X_axes = _invert_axes(Y_axes, ndim)

    #expand_dims for correct broadcasting with joint_pmf
    X_pmf = np.expand_dims(np.sum(joint_pmf, axis=Y_axes), Y_axes)
    Y_pmf = np.expand_dims(np.sum(joint_pmf, axis=X_axes), X_axes)
    return np.sum(joint_pmf[joint_pmf > 0] * np.log2((joint_pmf / (X_pmf * Y_pmf))[joint_pmf > 0]))


def test_MI_compare(_random_dataset):
    mi1 = MI(_random_dataset[0], _random_dataset[1])
    mi2 = _MI_2(_random_dataset[0], _random_dataset[1])

    assert mi1 >= 0
    assert np.abs(mi1 - mi2) < 1e-8


def test_MI_symmetric(_random_dataset):
    """Test I(X;Y) == I(Y;X)"""

    Y_axes = _random_dataset[1]
    X_axes = _invert_axes(Y_axes, _random_dataset[0].ndim)

    mi1 = MI(_random_dataset[0], Y_axes)
    mi2 = MI(_random_dataset[0], X_axes)

    assert mi1 >= 0
    assert np.abs(mi1 - mi2) < 1e-8


def test_MI_indentity1(_random_dataset):
    """Test I(X;Y) = H(X) - H(X|Y)"""

    Y_axes = _random_dataset[1]
    #X_axes = _invert_axes(Y_axes, _random_dataset[0].ndim)

    mi1 = MI(_random_dataset[0], Y_axes)
    mi2 = entropy(np.sum(_random_dataset[0], axis=Y_axes)) - conditional_entropy(_random_dataset[0], Y_axes)

    assert mi1 >= 0
    assert np.abs(mi1 - mi2) < 1e-8


def test_MI_indentity2(_random_dataset):
    """Test I(X;Y) = H(Y) - H(X|Y)"""

    Y_axes = _random_dataset[1]
    X_axes = _invert_axes(Y_axes, _random_dataset[0].ndim)

    assert X_axes != Y_axes

    mi1 = MI(_random_dataset[0], Y_axes)
    mi2 = entropy(np.sum(_random_dataset[0], axis=X_axes)) - conditional_entropy(_random_dataset[0], X_axes)

    assert mi1 >= 0
    assert np.abs(mi1 - mi2) < 1e-8


def test_MI_indentity3(_random_dataset):
    """Test I(X;Y) = H(Y) +  H(X) - H(X, Y)"""

    Y_axes = _random_dataset[1]
    X_axes = _invert_axes(Y_axes, _random_dataset[0].ndim)

    assert X_axes != Y_axes

    mi1 = MI(_random_dataset[0], Y_axes)
    mi2 = entropy(np.sum(_random_dataset[0], axis=X_axes)) +\
          entropy(np.sum(_random_dataset[0], axis=Y_axes)) -\
          entropy(_random_dataset[0])

    assert mi1 >= 0
    assert np.abs(mi1 - mi2) < 1e-8


def test_MI_indentity4(_random_dataset):
    """Test I(X;Y) = H(X,Y) -  H(X|Y) - H(Y|X)"""

    Y_axes = _random_dataset[1]
    X_axes = _invert_axes(Y_axes, _random_dataset[0].ndim)

    assert X_axes != Y_axes

    mi1 = MI(_random_dataset[0], Y_axes)
    mi2 = entropy(_random_dataset[0]) -\
          conditional_entropy(_random_dataset[0], X_axes) -\
          conditional_entropy(_random_dataset[0], Y_axes)

    assert mi1 >= 0
    assert np.abs(mi1 - mi2) < 1e-8