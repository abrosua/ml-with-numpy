import numbers
from typing import Optional, Union, List, Tuple

import numpy as np

from ..utils import check_is_fitted, issparse


class PCA:
    """
    Principal Component Analysis (PCA)

    This project only uses the LAPACK implementation of the full SVD,
    due to the limitation of using numpy only. Therefore it only accepts NON-SPARSE
    input. Compared to the scikit-learn's PCA, this project only cover the PCA
    with 'full' svd solver.

    Parameters
    ----------
    n_components:

    Attributes
    ----------
    components_: ndarray with (n_components, n_features) shape
        The principal axes, obtained from the eigenvectors of the input data.
        The components are sorted, in descending order, by`the eigenvalues or `explained_variance_``.

    explained_variance_: ndarray with (n_components,) shape
        The n_components largest eigenvalues of the covariance matrix of the input data, X.
        It represents the amount of variance explained by each of the selected components.

    explained_variance_ratio_ : ndarray with (n_components,) shape
        The 'explained_variance_' in percentage of the total explained variances.

    mean_ : ndarray with (n_features,) shape
        The mean value of each feature, based on the training data (X), `X.mean(axis=0)`..

    n_components_ : int or float
        The number of principal components. If the input is a greater than zero integer,
        the value will be used directly as the number of components.
        If the input is a float number between 0.0 to 1.0, the number of components
        will be estimated based on the number of explained variances.
        The n_components input will be the threshold. Otherwise it equals the lesser value of
        n_features and n_samples if n_components is None.

    n_features_ : int
        Number of features in the training data.

    n_samples_ : int
        Number of samples in the training data.
    """
    def __init__(self, n_components: Optional[Union[int, float]] = None):
        if n_components is not None:
            if type(n_components) in [int, float]:
                if not n_components > 0:
                    raise ValueError(f"The number of component(s) must be greater than ZERO!"
                                     f"The input was n_components={n_components}.")
            else:
                raise ValueError(f"Unknown number of components input occurs!"
                                 f"The input was n_components={n_components}.")

        self.n_components = n_components

    def fit(self, X: np.array, y=None):
        """
        Model fitting with X input

        Parameters
        ----------
        X: array with (n_samples, n_features) shape. Training data with,
           n_samples number of samples and n_features number of features.
        y: Ignored

        Returns
        -------
        self: object. The instance itself.
        """
        # Check if the input is a sparse array
        if issparse(X):
            raise TypeError("PCA does not support sparse array, please refer to scikit-learn for more solutions.")

        # Validate the input data
        # ***************

        n_samples, n_features = X.shape
        n_components = min(X.shape) - 1 if self.n_components is None else self.n_components

        if n_components < 0 or n_components > min(n_samples, n_features):
            raise ValueError(f"The number of components (n_components={n_components}) must be between 0 and"
                             f" {min(n_samples, n_features)}.")
        elif n_components >= 1:
            if not isinstance(n_components, numbers.Integral):
                raise ValueError(f"The number of components MUST be an INTEGER if it is greater or equal to 1 "
                                 f"(the input type was {type(n_components)}).")

        # Normalize the data
        self.mean_ = np.mean(X, axis=0)
        X = X - self.mean_

        # Calculating the eigenvalues and eigenvectors
        sigma = np.cov(X, rowvar=False)  # calculate the covariance matrix
        s, v = np.linalg.eig(sigma)  # eigen-decomposition

        # eigenvalue sorting with descending order
        idx = np.argsort(s)[::-1]
        v = v[:, idx]
        s = s[idx]
        s_ratio = s / s.sum()

        # Addressing the attributes
        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = v[:, :n_components].T
        self.explained_variance_ = s[:n_components]
        self.explained_variance_ratio_ = s_ratio[:n_components]

        return self

    def transform(self, X: np.array) -> np.array:
        """
        Applying dimensionality reduction to X input.

        The fitted model, based on the previous input, will project the X transform's input.

        Parameters
        ----------
        X: array with (n_samples, n_features) shape. New data with,
           n_samples number of samples and n_features number of features.

        Returns
        -------
        X_trans: array with (n_features, n_components) shape. Projected data by PCA components with,
           n_features number of features and n_components number of the PCA component(s).
        """
        check_is_fitted(self)

        # Validate the input data
        # ***** UNDER CONSTRUCTION *****
        if self.mean_ is not None:
            X = X - self.mean_
        X_trans = np.dot(X, self.components_.T)
        return X_trans

    def fit_transform(self, X: np.array) -> np.array:
        """
        Sequentially fit the model and applying dimensionality reduction to X.

        Parameters
        ----------
        X: array with (n_samples, n_features) shape. Act as both train and new data with,
           n_samples number of samples and n_features number of features.

        Return
        ------
        X_trans: array with (n_features, n_components) shape. Projected data by PCA components with,
           n_features number of features and n_components number of the PCA component(s).
        """
        X_trans = self.fit(X).transform(X)
        return X_trans
