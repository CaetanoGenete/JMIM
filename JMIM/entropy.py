import numpy as np
from numba import njit


@njit
def entropy(pmf: np.ndarray) -> float:
    """Calculates the entropy of a random variable X: H(X).

    Args:
        pmf (np.ndarray): The pmf of all the realisations of X

    Returns:
        float: The entropy of the the random variable X
    """

    #Flatten array to allow for numba to do its thing
    flat_pmf = pmf.ravel()
    return -np.sum(flat_pmf[flat_pmf > 0] * np.log2(flat_pmf[flat_pmf > 0]))


def _invert_axes(axes: tuple, ndim: int) -> tuple:
    """Computes the set minus operation (0,...,ndim-1)-axes"""

    # Mod ensures axes are in the interval [0, ndim) (Also allows for negative indices).
    return tuple(np.setdiff1d(np.arange(ndim), np.mod(axes, ndim)))


def conditional_entropy(joint_pmf: np.ndarray, Y_axes=(-1,)) -> float:
    """Compute the conditional entropy H(X_1,...,X_M|Y_1,...,Y_N), using the joint pmf of X_1,...,X_M,Y_1,...,Y_N.

    Args:
        joint_pmf (np.ndarray): Joint pmf of the RVs X_1,...,X_M,Y_1,...,Y_N (Each axis is a difference RV).
        Y_axes (tuple, optional): The axes of the RVs Y_1,...,Y_N. Defaults to (-1,).

    Returns:
        float: Conditional entropy
    """

    X_axes = _invert_axes(Y_axes, np.ndim(joint_pmf))
    # H(X_1,...,X_M,Y_1,...,Y_N) - H(Y_1,...,Y_N)
    return entropy(joint_pmf) - entropy(np.sum(joint_pmf, axis=X_axes))


def MI(joint_pmf: np.ndarray, Y_axes=(-1,)) -> float:
    """Compute the mutual information metric I(X_1,...,X_M;Y_1,...,Y_N), using the joint pmf of
    X_1,...,X_M,Y_1,...,Y_N.

    Args:
        joint_pmf (np.ndarray): Joint pmf of the RVs X_1,...,X_M,Y_1,...,Y_N (Each axis is a difference RV).
        Y_axes (tuple, optional): The axes of the RVs Y_1,...,Y_N. Defaults to (-1,).

    Returns:
        float: The value I(X_1,...,X_M;Y_1,...,Y_N).
    """

    # H(X_1,...,X_M) - H(X_1,...,X_M|Y_1,...,Y_N)
    return entropy(np.sum(joint_pmf, axis=Y_axes)) - conditional_entropy(joint_pmf, Y_axes)


def SR(joint_pmf: np.ndarray, Y_axes=(-1,)):
    """Compute the symmetrical relevance metric SR(X_1,...,X_M;Y_1,...,Y_N), using the joint pmf of
    X_1,...,X_M,Y_1,...,Y_N.

    Args:
        joint_pmf (np.ndarray): Joint pmf of the RVs X_1,...,X_M,Y_1,...,Y_N (Each axis is a difference RV).
        Y_axes (tuple, optional): The axes of the RVs Y_1,...,Y_N. Defaults to (-1,).

    Returns:
        float: The value I(X_1,...,X_M;Y_1,...,Y_N).
    """

    mi = MI(joint_pmf, Y_axes)
    # If H(X_1,...,X_M,Y_1,...,Y_N) is zero, then so is I(X_1,...,X_M;Y_1,...,Y_N). In this case, define SR = 0.
    return 0. if mi == 0 else (mi / entropy(joint_pmf))


def generate_pmf(data: np.ndarray, labels=None) -> np.ndarray:
    """Computes the pmf of the features (columns) of 'data', which must be uniquely labelled (For example by
    applying the 'JMI.preprocessing.label_data' function) with zero-based consecutive integers.

    Args:
        data (np.ndarray): A 2-dimensional array where each column contains consecutive integers. 
        labels (Any, optional): For each column, the mapping from the integer labels to the actual values.
        Defaults to None.

    Returns:
        np.ndarray: A numpy array with N-dimensions (Where N is the number of features), with each entry being
        the joint pmf of the particular labels appearing simultaneously.
    """

    assert np.ndim(data) == 2, "data must be a two dimensional numpy array"
    assert np.issubdtype(data.dtype, np.integer), "data must be a list of integral types, use"\
                                                  "JMIM.preprocessing.label_data to label"

    rows, cols = np.shape(data)

    # Stores the number of unique values for each feature (column)
    bins = None
    # If no labels are specified, calculate them
    if labels is None:
        bins = tuple(len(np.unique(column)) for column in data.T)
    else:
        assert len(labels) == cols, "Number of labels must equal number of features"
        bins = tuple(len(label) for label in labels)

    pmf = np.zeros(shape=bins)
    np.add.at(pmf, tuple(data.T), 1./rows)

    return pmf