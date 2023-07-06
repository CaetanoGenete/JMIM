import numpy as np

from collections.abc import Callable

from JMIM.entropy import MI, CMI, SR, entropy, conditional_entropy, _invert_axes, generate_pmf


def _bind_data(data: np.ndarray, labels, metric: Callable[[np.ndarray, tuple], float]) -> Callable[[tuple], float]:
    """Helper function, generates pmf from feature selection (new input) by binding labelled data"""

    return lambda features: metric(generate_pmf(data[:, features], [labels[i] for i in features]))


def _greedy_search(data: np.ndarray, k: int, setup: Callable[[list], int], criterion: Callable[[int, int], float], C=-1):
    """Greedy search selection function, at every step pick the feature with the GREATEST criterion"""

    F = list(_invert_axes((C,), np.shape(data)[1]))
    S = [F.pop(setup(F))]

    for _ in range(1, k):
        max_val = -np.inf
        max_index = None

        for i, f in enumerate(F):
            val = criterion(f, S[-1])

            if(val > max_val):
                max_val = val
                max_index = i

        S.append(F.pop(max_index))

    return S


def _MIM_common(data: np.ndarray, labels, k: int, C: int, criterion: Callable[[tuple], float]) -> list:
    """Common functionality between (x)MIM algorithms. IMPORTANT criterion args are (f_i, f_s, C)"""
    
    MI_metric = _bind_data(data, labels, MI)
    min_metric = np.empty(np.shape(data)[1])

    def _setup(F):
        min_metric[F] = [MI_metric((f, C)) for f in F]
        result = np.argmax(min_metric[F])

        #Reuse min_metric array
        min_metric[:] = np.inf
        return result

    def _criterion(f, s):
        min_metric[f] = min(min_metric[f], criterion((f, s, C)))
        return min_metric[f]

    return _greedy_search(data, k, _setup, _criterion, C)


def JMIM(data: np.ndarray, labels, k: int, C=-1, normalised=False) -> list:
    """Selects the k most significant features, based on the (N)JMIM algorithm.

    Args:
        data (np.ndarray): A two dimensional array of data where the columns represent the features.
        k (int): The number of features to select.
        labels (Any): The mapping from the integer labels in data to their true value. Defaults to None.
        C (int, optional): The output feature. Defaults to -1.
        normalised (bool, optional). If true, compute NJMIM instead. Defaults to False. 

    Returns:
        list: The k most significant features in order of significance.
    """

    return _MIM_common(data, labels, k, C, _bind_data(data, labels, SR if normalised else MI))


def CMIM(data: np.ndarray, labels, k: int, C=-1) -> list:
    """Selects the k most significant features, based on the CMIM algorithm.

    Args:
        data (np.ndarray): A two dimensional array of data where the columns represent the features.
        k (int): The number of features to select.
        labels (Any): The mapping from the integer labels in data to their true value. Defaults to None.
        C (int, optional): The output feature. Defaults to -1.

    Returns:
        list: The k most significant features in order of significance.
    """

    def _criterion(features: tuple) -> float:
        assert len(features) == 3
        return CMI(generate_pmf(data[:, features], [labels[i] for i in features]), (0,), (2,))

    return _MIM_common(data, labels, k, C, _criterion)


def MIFS(data: np.ndarray, labels, k: int, beta: float, C=-1) -> list:
    """Selects the k most significant features, based on the MIFS algorithm.

    Args:
        data (np.ndarray): A two dimensional array of data where the columns represent the features.
        labels (Any): The mapping from the integer labels in data to their true value. Defaults to None.
        k (int): The number of features to select.
        beta (float): The redudancy coefficient.
        C (int, optional): The output feature. Defaults to -1.

    Returns:
        list: The k most significant features in order of significance.
    """

    metric = _bind_data(data, labels, MI)
    min_metric = np.empty(np.shape(data)[1])

    def _setup(F):
        min_metric[F] = [metric((f, C)) for f in F]
        return np.argmax(min_metric[F])

    def _criterion(f, s):
        min_metric[f] -= beta * metric((f, s, C))
        return min_metric[f]
    
    return _greedy_search(data, k, _setup, _criterion, C)


def MIFS_U(data: np.ndarray, labels, k: int, beta: float, C=-1) -> list:
    """Selects the k most significant features, based on the MIFS-U algorithm.

    Args:
        data (np.ndarray): A two dimensional array of data where the columns represent the features.
        labels (Any): The mapping from the integer labels in data to their true value. Defaults to None.
        k (int): The number of features to select.
        beta (float): The redundancy coefficient.
        C (int, optional): The output feature. Defaults to -1.

    Returns:
        list: The k most significant features in order of significance.
    """

    MIfc = np.empty(np.shape(data)[1])
    Hf = np.empty(MIfc.shape)
    min_metric = np.empty(MIfc.shape)

    H_metric = _bind_data(data, labels, entropy)
    HC_metric = _bind_data(data, labels, conditional_entropy)
    # Rewrite MI to use cached Hf values
    MI_metric = lambda features: Hf[features[0]] - HC_metric(features)

    def _setup(F):        
        for fi in F:
            Hf[fi] = H_metric((fi, C))
            MIfc[fi] = MI_metric((fi, C))

        min_metric[:] = MIfc[:]
        return np.argmax(MIfc)

    def _criterion(f, s):
        min_metric[f] -= beta * MIfc[s] * MI_metric((s, f)) / Hf[s]
        return min_metric[f]
    
    return _greedy_search(data, k, _setup, _criterion, C)


def MRMR(data, labels, k: int, C=-1) -> list:
    """Selects the k most significant features, based on the MRMR algorithm.

    Args:
        data (np.ndarray): A two dimensional array of data where the columns represent the features.
        labels (Any): The mapping from the integer labels in data to their true value. Defaults to None.
        k (int): The number of features to select.
        C (int, optional): The output feature. Defaults to -1.

    Returns:
        list: The k most significant features in order of significance.
    """

    metric = _bind_data(data, labels, MI)
    # Select the f_i with the greatest metric
    u = np.empty(np.shape(data)[1])
    z = np.zeros(u.shape)

    def _setup(F):
        u[F] = [metric((f, C)) for f in F]
        return np.argmax(u[F])

    step = 0
    def _criterion(f, s):
        nonlocal step
        step += 1

        z[f] += metric((f, s))
        return u[f]-z[f]/step

    return _greedy_search(data, k, _setup, _criterion, C)