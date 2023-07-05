import numpy as np
import pytest

from JMIM.entropy import *
from JMIM.entropy import _invert_axes
from JMIM.preprocessing import label_data

from JMIM.JMIM import _JMIM_metric

@pytest.fixture(scope="module", params=list(range(6)))
def _random_dataset(request):
    np.random.seed(request.param)

    nfeatures = np.random.randint(3, 10)
    rows = np.random.randint(10_000, 50_000)

    data = np.random.randint(0, 10, size=(rows, nfeatures))
    return label_data(data)


@pytest.fixture(scope="module")
def _random_pmf(_random_dataset):
    data, labels = _random_dataset
    _, nfeatures = np.shape(data)

    pmf = generate_pmf(data, labels)
    assert np.abs(np.sum(pmf) - 1.) < 1e-10, "pmf must sum to 1!"
    assert np.alltrue(pmf >= 0), "all values of pmf must be non-negative!"

    Y_axes = tuple(range(1, np.random.randint(2, nfeatures+1)))

    return pmf, Y_axes


def _conditional_entropy_2(joint_pmf: np.ndarray, Y_axes=(-1,)) -> float:
    """For testing purposes. Identical computation to JMIM.entropy.conditional_entropy"""

    ndim = np.ndim(joint_pmf)

    #Ensure axes are in the interval [0, ndim) (Also allows for negative indices).
    Y_axes = np.mod(Y_axes, ndim)
    X_axes = _invert_axes(Y_axes, ndim)

    # expand_dims for correct broadcasting with joint_pmf
    Y_pmf = np.expand_dims(np.sum(joint_pmf, axis=X_axes), X_axes)
    # (May throw divide by zero warning)
    return -np.sum(joint_pmf[joint_pmf > 0] * np.log2((joint_pmf / Y_pmf)[joint_pmf > 0]))


def test_entropy_compare(_random_pmf):
    """Compare result of the explicit summation of conditional entropy"""

    pmf, Y_axes = _random_pmf

    e1 = conditional_entropy(pmf, Y_axes)
    e2 = _conditional_entropy_2(pmf, Y_axes)

    assert np.abs(e1 - e2) < 1e-8


def _MI_2(joint_pmf: np.ndarray, Y_axes=(-1,)) -> float:
    """For testing purposes. Identical computation to JMIM.entropy.MI"""

    ndim = np.ndim(joint_pmf)

    #Ensure axes are in the interval [0, ndim) (Also allows for negative indices).
    Y_axes = tuple(np.mod(Y_axes, ndim))
    X_axes = _invert_axes(Y_axes, ndim)

    #expand_dims for correct broadcasting with joint_pmf
    X_pmf = np.expand_dims(np.sum(joint_pmf, axis=Y_axes), Y_axes)
    Y_pmf = np.expand_dims(np.sum(joint_pmf, axis=X_axes), X_axes)
    return np.sum(joint_pmf[joint_pmf > 0] * np.log2((joint_pmf / (X_pmf * Y_pmf))[joint_pmf > 0]))


def test_MI_compare(_random_pmf):
    """Compare result of the explicit summation of MI"""

    pmf, Y_axes = _random_pmf

    mi1 = MI(pmf, Y_axes)
    mi2 = _MI_2(pmf, Y_axes)

    assert mi1 >= 0
    assert np.abs(mi1 - mi2) < 1e-8


def test_MI_symmetric(_random_pmf):
    """Test I(X;Y) == I(Y;X)"""

    pmf, Y_axes = _random_pmf
    X_axes = _invert_axes(Y_axes, np.ndim(pmf))

    mi1 = MI(pmf, Y_axes)
    mi2 = MI(pmf, X_axes)

    assert mi1 >= 0
    assert np.abs(mi1 - mi2) < 1e-8


def test_MI_indentity1(_random_pmf):
    """Test I(X;Y) = H(X) - H(X|Y)"""

    pmf, Y_axes = _random_pmf

    mi1 = MI(pmf, Y_axes)
    mi2 = entropy(np.sum(pmf, axis=Y_axes)) - conditional_entropy(pmf, Y_axes)

    assert mi1 >= 0
    assert np.abs(mi1 - mi2) < 1e-8


def test_MI_indentity2(_random_pmf):
    """Test I(X;Y) = H(Y) - H(X|Y)"""

    pmf, Y_axes = _random_pmf
    X_axes = _invert_axes(Y_axes, np.ndim(pmf))

    assert X_axes != Y_axes

    mi1 = MI(pmf, Y_axes)
    mi2 = entropy(np.sum(pmf, axis=X_axes)) - conditional_entropy(pmf, X_axes)

    assert mi1 >= 0
    assert np.abs(mi1 - mi2) < 1e-8


def test_MI_indentity3(_random_pmf):
    """Test I(X;Y) = H(Y) +  H(X) - H(X, Y)"""

    pmf, Y_axes = _random_pmf
    X_axes = _invert_axes(Y_axes, np.ndim(pmf))

    assert X_axes != Y_axes

    mi1 = MI(pmf, Y_axes)
    mi2 = entropy(np.sum(pmf, axis=X_axes)) +\
          entropy(np.sum(pmf, axis=Y_axes)) -\
          entropy(pmf)

    assert mi1 >= 0
    assert np.abs(mi1 - mi2) < 1e-8


def test_MI_indentity4(_random_pmf):
    """Test I(X;Y) = H(X,Y) -  H(X|Y) - H(Y|X)"""

    pmf, Y_axes = _random_pmf
    X_axes = _invert_axes(Y_axes, np.ndim(pmf))

    assert X_axes != Y_axes

    mi1 = MI(pmf, Y_axes)
    mi2 = entropy(pmf) -\
          conditional_entropy(pmf, X_axes) -\
          conditional_entropy(pmf, Y_axes)

    assert mi1 >= 0
    assert np.abs(mi1 - mi2) < 1e-8


def test_joint_MI_ineq(_random_dataset):
    data, labels = _random_dataset
    _, nfeatures = np.shape(data)

    JMI = _JMIM_metric(data, labels, MI)

    for i in range(nfeatures-1):
        for j in range(i+1):
            jmi = JMI((i, j, -1))
            
            assert jmi >= JMI((i, -1)) and jmi >= JMI((j, -1))