
import numba as nb
import numpy as np


@nb.njit(parallel=True, fastmath=True)
def pairwise_distance(X1, X2, D):
    x1 = np.sum(np.square(X1), axis=1)
    x2 = np.sum(np.square(X2), axis=1)
    np.dot(X1, X2.T, D)
    D *= -2
    x3 = x1.reshape(x1.size, 1)
    np.add(D, x3, D)
    np.add(D, x2, D)
    np.sqrt(D, D)
