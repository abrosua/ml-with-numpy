from typing import Optional, Union

import numpy as np

from ._math import euclidean_distance, manhattan_distance
from ..utils import check_is_fitted


class KNeighborsRegressor:
    """
    Performing regression using k-nearest neighbors model.

    k-NN is a non-parametric ML supervised learning model, thus unlike logistic regressions or neural network
    this model does NOT have a parameters to trained (i.e., weights and biases).

    Parameters
    ----------
    k:  int. The default is 2
        The number of nearest neighbors.

    p:  int. The default is 2.
        The method to calculate the data points distance
        p = 1 for l1 distance or manhattan distance
        p = 2 for l2 distance or euclidean distance

    Attributes
    ------

    """
    def __init__(self, k: int = 2, p: int = 2):
        min_k = 1
        if type(k) is not int or k <= min_k:
            raise ValueError(f"The number of nearest neighbors (k) has to be an integer "
                             f"that is larger than {min_k}! The input was k={k} with {type(k)} type.")
        if p not in [1, 2]:
            raise ValueError(f"The current model only supports either the l1, manhattan, distance (p=1) "
                             f"or l2, euclidean, distance (p=2)! The input was p={p}.")

        self.k = k
        self.p = p
        self.calc_distance = manhattan_distance if p == 1 else euclidean_distance

    def fit(self, X: np.array, y: np.array):
        """
        Fitting the K-NN regression model to the training dataset

        Parameters
        ----------
        X:  ndarray with (n_train_samples, n_features) shape. Training data with,
            n_train_samples number of samples and n_features number of features.
        y:  ndarray with (n_train_samples,) shape. The training data label/target.

        Return
        ------
        self:   KNeighborsRegressor
                The fitted k-NN regression model, based on X and y.
        """
        # Storing the training dataset
        self.X_ = X
        self.y_ = y

        # k-NN is a non-parametric model, it does NOT have parameters (i.e., weights and biases) to train!
        return self

    def predict(self, X: np.array) -> np.array:
        """
        Predict the target/label of the testing dataset,
        by performing the k-NN regression fitted by the training dataset.

        Parameters
        ----------
        X:  ndarray with (n_test_samples, n_features) shape. Testing data with,
            n_test_samples number of samples and n_features number of features.
            The target/label of this dataset will be predicted later.

        Return
        ------
        y_pred: ndarray with (n_test_samples,) shape.
                The predicted target/label of the testing set.
        """
        check_is_fitted(self)  # Check if self.fit is already performed before
        y_pred_list = []  # Initialize the prediction

        for i in range(len(X)):  # Using loop to avoid memory overflow if the array is too large
            # Computing the euclidean distance of each feature (X_i) to the testing set (X_ij)
            distance = self.calc_distance(self.X_, X[i])
            distance_sorted_id = np.argsort(distance)

            # Predicting the new label (y_pred) based on the average of the k-nearest y value
            y_pred_i = self.y_[distance_sorted_id][:self.k].mean()
            y_pred_list.append(y_pred_i)

        y_pred = np.array(y_pred_list)
        return y_pred

    def fit_predict(self, X_train, y_train, X_test) -> np.array:
        """
        Sequentially predict the target/label of the testing data after fitting the model
        to the training dataset.

        Parameters
        ----------
        X_train:    ndarray with (n_train_samples, n_features) shape. Training data with,
                    n_train_samples number of samples and n_features number of features.

        y_train:    ndarray with (n_train_samples,) shape. The training data label/target.

        X_test:     ndarray with (n_test_samples, n_features) shape. Testing data with,
                    n_test_samples number of samples and n_features number of features.
                    The target/label of this dataset will be predicted later.

        Return
        ------
        y_pred:     ndarray with (n_test_samples,) shape.
                    The predicted target/label of the testing set.
        """
        y_pred = self.fit(X_train, y_train).predict(X_test)
        return y_pred
