import numpy as np

from JMIM.entropy import MI, _invert_axes, label_data, generate_pmf


def _JMI(data, labels, axes):
    """Helper function to compute I(f_1,...,f_{N-1};F_N) where axes=f_1,...,f_{N-1},F_N"""

    bins = tuple(len(labels[i]) for i in axes)
    return MI(generate_pmf(data, bins, axes=axes))


def JMIM(data, k, labels=None, C=-1):
    """Computes the JMIM

    Args:
        data (_type_): _description_
        k (_type_): _description_
        labels (_type_, optional): _description_. Defaults to None.
        C (int, optional): _description_. Defaults to -1.

    Returns:
        _type_: _description_
    """

    _, nfeatures = np.shape(data)
    assert k < nfeatures, "k cannot be greater or equal to the number of features"

    # If no labels are specified, assign labels to data
    if(labels is None):
        data, labels = label_data(data)
    else:
        assert nfeatures == len(labels), "Number of label sets doesn't match number of features"

    # F initially contains every feature (except C)
    F = list(_invert_axes((C,), nfeatures))
    S = []

    # Select the f_i with the greatest I(f_i;C)
    max_index = np.argmax([_JMI(data, labels, (fi, C)) for fi in F])
    S.append(F.pop(max_index))

    min_JMIs = [np.inf] * len(F)
    # Every iteration, S grows by 1. Stop when |S|=k.
    for _ in range(1, k):
        # Recompute min(I(f_i,f_s;C)) now that there is a new feature in S
        for i, min_val in enumerate(min_JMIs):
            min_JMIs[i] = min(min_val, _JMI(data, labels, (F[i], S[-1], C)))

        max_index = np.argmax(min_JMIs)
        S.append(F.pop(max_index))
        min_JMIs.pop(max_index)

    return S