import numpy as np

from collections.abc import Callable

from JMIM.entropy import MI, _invert_axes, generate_pmf, SR
from JMIM.preprocessing import label_data


def _prep_data(data, k: int, labels, C: int) -> tuple:
    """Helper function, labels data (if not already labelled) and initialises arrays"""

    _, nfeatures = np.shape(data)
    assert k < nfeatures, "k must be less than the number of features"

    # If no labels are specified, assign labels to data
    if(labels is None):
        data, labels = label_data(data)
    else:
        assert len(labels) == nfeatures, "Number of label sets doesn't match number of features"

    # F initially contains every feature (except C)
    F = list(_invert_axes((C,), nfeatures))
    return data, labels, F, []


def _JMIM_metric(data: np.ndarray, labels, metric: Callable[[np.ndarray, tuple], float]) -> Callable[[tuple], float]:
    return lambda features: metric(generate_pmf(data[:, features], [labels[i] for i in features]))


def JMIM(data, k: int, labels=None, C=-1, normalised=False) -> list:
    """Selects the k most significant features, based on the (N)JMIM algorithm.

    Args:
        data (Any): A two dimensional array of data where the columns represent the features.
        k (int): The number of features to select.
        labels (Any, optional): The mapping from the integer labels in data to their true value. Defaults to None.
        C (int, optional): The output feature. Defaults to -1.
        normalised (bool, optional). If true, compute NJMIM instead. Defaults to False. 

    Returns:
        list: The k most significant features in order of significance.
    """

    data, labels, F, S = _prep_data(data, k, labels, C)
    # NJMIM is identical to JMIM, but using SR instead of MI as the metric
    metric = _JMIM_metric(data, labels, SR if normalised else MI)

    # Select the f_i with the greatest metric
    min_metrics = np.array([metric((fi, C)) for fi in F])
    S.append(F.pop(np.argmax(min_metrics)))
    
    #Unfortunately, I(f_i, f_j;C) >= I(f_i; C), must reset.
    min_metrics[:] = np.inf
    # Every iteration, S grows by 1. Stop when |S|=k.
    for _ in range(1, k):
        for f in F:
            min_metrics[f] = min(min_metrics[f], metric((f, S[-1], C)))
        
        S.append(F.pop(np.argmax(min_metrics[F])))

    return S


def MIFS(data, k: int, labels=None, C=-1) -> list:
    """Selects the k most significant features, based on the MIFS algorithm.

    Args:
        data (Any): A two dimensional array of data where the columns represent the features.
        k (int): The number of features to select.
        labels (Any, optional): The mapping from the integer labels in data to their true value. Defaults to None.
        C (int, optional): The output feature. Defaults to -1.

    Returns:
        list: The k most significant features in order of significance.
    """

    data, labels, F, S = _prep_data(data, k, labels, C)
    metric = _JMIM_metric(data, labels, MI)

    # Select the f_i with the greatest metric
    min_metrics = np.array([metric((fi, C)) for fi in F])
    S.append(F.pop(np.argmax(min_metrics)))
    
    # Every iteration, S grows by 1. Stop when |S|=k.
    for _ in range(1, k):
        for f in F:
            min_metrics[f] - metric((f, S[-1], C))
        
        S.append(F.pop(np.argmax(min_metrics[F])))

    return S