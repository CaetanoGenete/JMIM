import numpy as np
from numba import njit


@njit
def entropy(pmf) -> float:
    """Calculates the entropy of a random variable X: H(X).

    Args:
        pmf (np.array): The pmf of all the realisations of X

    Returns:
        float: The entropy of the the random variable X
    """

    #Flatten array to allow for numba to do its thing
    flat_pmf = pmf.ravel()
    return -np.sum(flat_pmf[flat_pmf > 0] * np.log2(flat_pmf[flat_pmf > 0]))


def _invert_axes(axes, ndim) -> tuple:
    #Mod ensures axes are in the interval [0, ndim) (Also allows for negative indices).
    return tuple(np.setdiff1d(np.arange(ndim), np.mod(axes, ndim)))


def conditional_entropy(joint_pmf, Y_axes=(-1,)) -> float:
    """Compute the conditional entropy H(X_1,...,X_M|Y_1,...,Y_N), using the joint pmf of X_1,...,X_M,Y_1,...,Y_N.

    Args:
        joint_pmf (np.array): Joint pmf of the RVs X_1,...,X_M,Y_1,...,Y_N (Each axis is a difference RV).
        Y_axes (tuple, optional): The axes of the RVs Y_1,...,Y_N. Defaults to (-1,).

    Returns:
        float: Conditional entropy
    """

    X_axes = _invert_axes(Y_axes, np.ndim(joint_pmf))
    # H(X_1,...,X_M,Y_1,...,Y_N) - H(Y_1,...,Y_N)
    return entropy(joint_pmf) - entropy(np.sum(joint_pmf, axis=X_axes))


def MI(joint_pmf, Y_axes=(-1,)) -> float:
    """Compute the mutual information metric I(X_1,...,X_M;Y_1,...,Y_N), using the joint pmf of
    X_1,...,X_M,Y_1,...,Y_N.

    Args:
        joint_pmf (np.array): Joint pmf of the RVs X_1,...,X_M,Y_1,...,Y_N (Each axis is a difference RV).
        Y_axes (tuple, optional): The axes of the RVs Y_1,...,Y_N. Defaults to (-1,).

    Returns:
        float: Mutual information
    """

    # H(X_1,...,X_M) - H(X_1,...,X_M|Y_1,...,Y_N)
    return entropy(np.sum(joint_pmf, axis=Y_axes)) - conditional_entropy(joint_pmf, Y_axes)


def generate_pmf(data, labels=None, axes=None) -> np.ndarray:
    """Computes the pmf of the features (columns) of 'data', which must be uniquely labelled (For example by
    applying the 'label_data' function).

    Args:
        data (_type_): _description_
        labels (_type_, optional): _description_. Defaults to None.
        axes (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    assert np.ndim(data) == 2, "data must be a two dimensional numpy array"
    assert np.issubdtype(data.dtype, np.integer), "data must be a list of integral types, use label_data to convert"

    rows, cols = np.shape(data)

    # By default, calculate joint pmf for all features
    if axes is None:
        axes = tuple(range(cols))

    # Stores the number of unique values for each feature (column)
    bins = None
    # If no labels are specified, calculate them
    if labels is None:
        bins = tuple(len(set(column)) for column in data[:, axes].T)
    else:
        assert len(labels) == len(axes), "Size of bin counts tuple must equal number of axes"
        bins = tuple(len(label) for label in labels)

    pmf = np.zeros(shape=bins)
    np.add.at(pmf, tuple(data[:, axes].T), 1./rows)

    return pmf