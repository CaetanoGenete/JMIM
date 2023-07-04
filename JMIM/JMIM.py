import numpy as np

from JMIM.entropy import MI, _invert_axes


def _reduce_joint_pmf(joint_pmf, axes):
    """Computes the joint pmf for a smaller subset of features (axes). Features are relabelled to be
    consecutive whilst preserving their order"""

    return np.sum(joint_pmf, axis=_invert_axes(axes, joint_pmf.ndim))


def JMIM(joint_pmf, k):
    assert k < joint_pmf.ndim, "Selecting too many features!"

    F = list(range(joint_pmf.ndim-1))
    S = []

    max_index = np.argmax([MI(_reduce_joint_pmf(joint_pmf, (fi, -1))) for fi in F])
    S.append(F.pop(max_index))

    for _ in range(1, k):
        max_index = np.argmax([np.min([MI(_reduce_joint_pmf(joint_pmf, (fi, fs, -1))) for fs in S]) for fi in F])
        S.append(F.pop(max_index))

    return S


def _JMIM_2(joint_pmf, k):
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