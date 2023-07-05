import numpy as np
import pytest

from JMIM.preprocessing import label_data


@pytest.fixture(scope="module", params=list(range(20)))
def _random_dataset(request):
    np.random.seed(request.param)

    nfeatures = np.random.randint(3, 30)
    rows = np.random.randint(10_000, 50_000)

    return np.random.randn(rows, nfeatures)


@pytest.fixture(scope="module", params=list(range(20)))
def _random_choice_dataset(request):
    np.random.seed(request.param)

    nfeatures = np.random.randint(3, 30)
    rows = np.random.randint(10_000, 50_000)

    random_numbers = np.random.rand(np.random.randint(5, 20))
    return np.random.choice(random_numbers, size=(rows, nfeatures))


def test_relabel_unique(_random_choice_dataset):
    """Test labelling the data and then acquiring the original data from the labelled"""

    rows, nfeatures = np.shape(_random_choice_dataset)

    labelled_data, labels = label_data(_random_choice_dataset)
    relabelled_data = np.empty((rows, nfeatures))

    for i in range(nfeatures):
        relabelled_data[:, i] = np.vectorize(lambda x: labels[i][x])(labelled_data[:, i])

    assert np.alltrue(relabelled_data == _random_choice_dataset)


def _test_label_efd_feature(feature, labelled_feature, label, bins):
    """Helper function, pulls out common functionality in np.ndarray and pd.DataFrama test variants"""

    counts = np.bincount(labelled_feature)
    expected_freq = len(labelled_feature)/float(bins)

    assert len(counts) == bins
    assert np.alltrue(np.abs(counts - expected_freq) <= 1.)
    
    nplabel = np.array(label)
    assert np.alltrue(nplabel[labelled_feature, 0] <= feature) and\
           np.alltrue(feature <= nplabel[labelled_feature, 1])


def test_label_efd(_random_dataset):
    bins = 10
    labelled_data, labels = label_data(_random_dataset, method="efd", bins=bins)

    for triplet in zip(_random_dataset.T, labelled_data.T, labels):
        _test_label_efd_feature(*triplet, bins)


def _test_label_ewd_feature(feature, labelled_feature, label, bins):
    assert len(np.unique(labelled_feature)) == bins

    nplabel = np.array(label)
    widths = [b-a for (a, b) in nplabel][1:-1]

    # Widths of intervals (except at the ends) should all be equal
    assert np.allclose(widths[0], widths, 1e-8)

    assert np.alltrue(nplabel[labelled_feature, 0] <= feature) and\
            np.alltrue(feature <= nplabel[labelled_feature, 1])


def test_label_ewd(_random_dataset):
    bins = 10
    labelled_data, labels = label_data(_random_dataset, method="ewd", bins=bins)

    for triplet in zip(_random_dataset.T, labelled_data.T, labels):
        _test_label_ewd_feature(*triplet, bins)
        
# ----------------------------------------------------------- Pandas DataFrame compatibility tests ----------------------------------------------------------- #

try:
    import pandas as pd


    @pytest.fixture(scope="module")
    def _random_pandas_dataset(_random_dataset):
        return pd.DataFrame(_random_dataset)
    

    def test_label_ewd_pandas(_random_pandas_dataset):
        assert type(_random_pandas_dataset) is pd.DataFrame

        bins = 10
        labelled_data, labels = label_data(_random_pandas_dataset, method="ewd", bins=bins)

        for feature, *double in zip(_random_pandas_dataset, labelled_data.T, labels):
            _test_label_ewd_feature(_random_pandas_dataset[feature], *double, bins)


    def test_label_efd_pandas(_random_pandas_dataset):
        assert type(_random_pandas_dataset) is pd.DataFrame

        bins = 10
        labelled_data, labels = label_data(_random_pandas_dataset, method="efd", bins=bins)

        for feature, *double in zip(_random_pandas_dataset, labelled_data.T, labels):
            _test_label_efd_feature(_random_pandas_dataset[feature], *double, bins)
    
    
    def test_relabel_unique_pandas(_random_choice_dataset):
        test_relabel_unique(pd.DataFrame(_random_choice_dataset))

except:
    print("[INFO] Cannot import pandas, disabling tests.")