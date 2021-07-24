import numpy as np


def euclidean_distance(x1, x2):
    d = np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
    return d


def manhattan_distance(x1, x2):
    d = np.sum(np.abs(x1 - x2), axis=1)
    return d
