import numpy as np


def check_invertible_array(X: np.array) -> bool:
    """
    Check if the input array is invertible based on its rank
    Parameters
    ----------
    X:  np.array with (n_samples, n_features) shape

    Returns
    -------
    boolean, True if X is invertible, False if NOT.
    """
    assert len(X.shape) == 2  # Has to be 2-D array
    assert np.divide(*X.shape) == 1  # Has to be a square array/matrix

    n_rank_raw, _ = X.shape
    n_rank_real = np.linalg.matrix_rank(X)

    return True if int(n_rank_raw) == int(n_rank_real) else False

def feature_norm(X: np.array, axis: int = 0) -> np.array:
    """
    Feature normalization.
    X_ = (X - mean(X)) / stdev(X)

    Parameters
    ----------
    X:  np.array with (n_samples, n_features) shape
    axis: int, which axis to perform the normalization too (features' axis).

    Returns
    -------
    X_new:  np.array with (n_samples, n_features) shape.
            The normalized X input based on its feature!
    """
    X_new = (X - np.mean(X, axis=axis)) / np.std(X, axis=axis)
    return X_new

def normal_eq(X: np.array, y: np.array) -> np.array:
    """
    Calculating normal equation to solve linear regression.

    Parameters
    ----------
    X: np.array with (n_samples, n_features) shape
    y: np.array with (n_samples,) shape

    Returns
    -------
    theta:  np.array with (n_features,).
            The trained parameters for the linear regression.
    """
    A = np.dot(X.T, X)
    theta = np.dot(np.linalg.pinv(A), np.dot(X.T, y))  # inv(A) dot X.T dot y
    return theta
