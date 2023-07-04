import numpy as np
import pytest

from JMIM.preprocessing import label_data


@pytest.fixture(scope="module", params=list(range(20)))
def _random_dataset(request):
    np.random.seed(request.param)

    nfeatures = np.random.randint(3, 30)
    rows = np.random.randint(10_000, 50_000)

    random_numbers = np.random.rand(np.random.randint(5, 20))
    data = np.random.choice(random_numbers, size=(rows, nfeatures))
    return data, *label_data(data)


def test_relabel(_random_dataset):
    data, labelled_data, labels = _random_dataset
    rows, cols = np.shape(_random_dataset[0])

    relabelled_data = np.zeros((rows, cols), dtype=data.dtype)

    for i in range(cols):
        relabelled_data[:, i] = np.vectorize(lambda x: labels[i][x])(labelled_data[:, i])

    assert np.alltrue(relabelled_data == data)