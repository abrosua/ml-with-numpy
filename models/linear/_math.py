import numpy as np


def check_invertible_array(X) -> bool:
    assert len(X.shape) == 2  # Has to be 2-D array
    assert np.divide(*X.shape) == 1  # Has to be a square array/matrix

    n_rank_raw, _ = X.shape
    n_rank_real = np.linalg.matrix_rank(X)

    return True if int(n_rank_raw) == int(n_rank_real) else False

def feature_norm(X: np.array, axis: int = 0) -> np.array:
    X_new = (X - np.mean(X, axis=axis)) / np.std(X, axis=axis)
    return X_new
