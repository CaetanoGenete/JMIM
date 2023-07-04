import numpy as np
import pytest

from JMIM.entropy import *
from JMIM.entropy import _invert_axes
from JMIM.preprocessing import label_data
from JMIM.JMIM import JMIM


@pytest.fixture(scope="module", params=list(range(6)))
def _random_dataset(request):
    np.random.seed(request.param)

    nfeatures = np.random.randint(3, 10)
    rows = np.random.randint(10_000, 50_000)

    data = np.random.randint(0, 10, size=(rows, nfeatures))
    labelled_data, labels = label_data(data)

    pmf = generate_pmf(labelled_data, labels)

    assert np.abs(np.sum(pmf) - 1.) < 1e-10, "pmf must sum to 1!"
    assert np.alltrue(pmf >= 0), "all values of pmf must be non-negative!"

    return pmf, data


def _reduce_joint_pmf(joint_pmf, axes):
    """Computes the joint pmf for a smaller subset of features (axes). Features are relabelled to be
    consecutive whilst preserving their order"""

    return np.sum(joint_pmf, axis=_invert_axes(axes, joint_pmf.ndim))


def _JMIM_2(joint_pmf, k):
    """For testing purposes. Identical computation to JMIM.JMIM.JMIM"""

    assert k < joint_pmf.ndim, "Selecting too many features!"

    # Omit last feature since this is C
    F = list(range(joint_pmf.ndim-1))
    S = []

    max_index = np.argmax([MI(_reduce_joint_pmf(joint_pmf, (fi, -1))) for fi in F])
    S.append(F.pop(max_index))

    for _ in range(1, k):
        max_index = np.argmax([np.min([MI(_reduce_joint_pmf(joint_pmf, (fi, fs, -1))) for fs in S]) for fi in F])
        S.append(F.pop(max_index))

    return S


def _JMIM_3(joint_pmf, k):
    """For testing purposes. Identical computation to JMIM.JMIM.JMIM"""

    # Omit last feature since this is C
    F = list(range(joint_pmf.ndim-1))
    S = []

    # Select the f_i with the greatest I(f_i;C)
    max_index = np.argmax([MI(_reduce_joint_pmf(joint_pmf, (fi, -1))) for fi in F])
    S.append(F.pop(max_index))

    mins = [np.inf for _ in F]
    for _ in range(1, k):
        # Recompute min(I(f_i,f_s;C)) now that there is a new feature in S
        for i, min_val in enumerate(mins):
            mins[i] = min(min_val, MI(_reduce_joint_pmf(joint_pmf, (F[i], S[-1], -1))))

        max_index = np.argmax(mins)
        S.append(F.pop(max_index))
        mins.pop(max_index)

    return S


@pytest.mark.parametrize("k_frac", [0., 0.25, 0.5, 0.75])
def test_JMIM_1(_random_dataset, k_frac):
    """Compare results of _JMIM_2 implementation with JMIM"""

    pmf, data = _random_dataset

    ndim = np.ndim(data)
    k = min(max(1, int(k_frac * ndim)), ndim-1)

    result1 = JMIM(data, k)
    result2 = _JMIM_2(pmf, k)

    assert result1 == result2
    assert len(np.unique(result1)) == len(result1)


@pytest.mark.parametrize("k_frac", [0., 0.25, 0.5, 0.75])
def test_JMIM_2(_random_dataset, k_frac):
    """Compare results of _JMIM_3 implementation with JMIM"""

    pmf, data = _random_dataset

    ndim = np.ndim(data)
    k = min(max(1, int(k_frac * ndim)), ndim-1)

    result1 = JMIM(data, k)
    result2 = _JMIM_3(pmf, k)

    assert result1 == result2
    assert len(np.unique(result1)) == len(result1)