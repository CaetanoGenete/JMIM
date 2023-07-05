import numpy as np
from functools import singledispatch

def _unique_labelling(data, _) -> tuple:
    """Assign a unique integer label to every value in data"""

    #Using set here to allow for non-numeric types
    value_to_label = {value:j for j, value in enumerate(set(data))}
    
    labelled_data = np.vectorize(lambda x: value_to_label[x])(data)
    label = list(value_to_label.keys())

    return labelled_data, label


def _digitise_split_bins(data, bins) -> tuple:
    # Ensure all values on the number line are covered by the bins
    bins[0] = -np.inf
    bins[-1] = np.inf

    labelled_data = np.digitize(data, bins)-1
    label = [(left, right) for left, right in zip(bins[:-1], bins[1:])]

    return labelled_data, label


def _equal_width_binning(data, bins_count: int) -> tuple:
    """Create equally spaced intervals, uniquely labelled by zero-based increasing integers."""

    bins = np.linspace(np.min(data), np.max(data), bins_count+1)
    return _digitise_split_bins(data, bins)


def _equal_freq_binning(data, bins_count: int) -> tuple:
    """Create intervals containing an equal number of elements, uniquely labelled by zero-based increasing integers."""

    bins = np.interp(
        np.linspace(0, len(data), bins_count+1),
        np.arange(len(data)),
        np.sort(data))
    
    return _digitise_split_bins(data, bins)


def _index_maybe_scalar(array, index: int):
    return array if array is None or np.isscalar(array) else array[index]


@singledispatch
def _label_data_iter(data):
    return np.transpose(data)

# Pandas DataFrame specialisation
try:
    import pandas as pd

    @_label_data_iter.register
    def _(data: pd.DataFrame):
        return data.T.iloc

except ImportError:
    pass


_label_data_method_map = {
    "unique": _unique_labelling,
    "ewd": _equal_width_binning,
    "efd": _equal_freq_binning
}
def label_data(data, method="unique", bins=None) -> tuple:
    """For each column in 'data', assigns a unique id to each value.

    Args:
        data (Any): A 2-dimensional array.

    Returns:
        tuple: A numpy array with the same dimensions as 'data' of integer type, with every value in each
        column uniquely labelled from 0 to N; and a map from these labels to their original values as a ragged
        two-dimensional list.
    """

    # Map from label to value for each feature
    labelled_data = np.zeros(np.shape(data), dtype=np.uint8)
    labels = []

    # Uniquely label the values of each feature (column) from 0,...
    for i, feature in enumerate(_label_data_iter(data)):
        method_i = _index_maybe_scalar(method, i).lower()
        bins_i = _index_maybe_scalar(bins, i)
        
        assert method in _label_data_method_map.keys(), "Unknown method!"

        labelled_data[:, i], label = _label_data_method_map[method_i](feature, bins_i)
        labels.append(label)

    return labelled_data, labels