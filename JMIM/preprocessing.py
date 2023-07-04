import numpy as np

def label_data(data) -> tuple:
    """For each column in 'data', assigns a unique id to each value.

    Args:
        data (Any): A 2-dimensional array

    Returns:
        tuple: A numpy array with the same dimensions as 'data' of integer type, with every value in each
        column uniquely labelled from 0 to N; and a map from these labels to their original values as a ragged
        two-dimensional list.
    """

    assert np.ndim(data) == 2, "data must be a two dimensional array"

    _, cols = np.shape(data)
    # Map from label to value for each feature
    labelled_data = np.zeros(np.shape(data), dtype=np.uint8)
    labels = []
    
    # Uniquely label the values of each feature (column) from 0,...
    for i in range(cols):
        #Using 'set' instead of 'np.unique' to allow for non-numeric data-types
        value_to_label = {value:j for j, value in enumerate(set(data[:, i]))}
        labelled_data[:, i] = np.vectorize(lambda x: value_to_label[x])(data[:, i])

        labels.append(list(value_to_label.keys()))

    return labelled_data, labels