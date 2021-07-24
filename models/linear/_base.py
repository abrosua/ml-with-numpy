from typing import Optional, Union
import warnings

import numpy as np

from ._math import check_invertible_array, feature_norm
from ..utils import check_is_fitted, issparse


class LinearRegression:
    """
    The Linear Regression implementation using NumPy.

    The Linear Regression model try to fit a linear model with weights coefficient,
    and bias intercept, by minimizing the error differences between the prediction
    and the target label. Unlike the scikit-learn implementation,
    this model automatically perform data centering onto the input dataset.

    Parameters
    ----------
    a

    Attributes
    ----------
    coef_:  ndarray with (n_features,) or (n_targets, n_features) shape.
            The trained parameters (model's coefficient) of the Linear Regression.
            n_features is the number of parsed features while
            n_targets is the number of targets used if the target label (y) is in 2D.
            i.e., y is an ndarray with (n_samples, n_target) shape

    intercept_: float or ndarray with (n_targets,)
                Model's intercept point.
    """
    def __init__(self, alpha: Optional[float] = None, n_iter: Optional[int] = None, cost: int = 2, verbose: int = 1):
        # Checking the inputs for cost function option
        if cost not in [1, 2]:
            raise ValueError(f"Unknown cost function option (), choose between 1 or 2 only!")

        # Checking the inputs for alpha and n_iter
        if alpha is None and n_iter is None:
            self._use_grad = False
            print("Solving the Linear Regression using NORMAL EQUATION method!") if verbose else None

        else:
            self._use_grad = True
            print("Solving the Linear Regression using GRADIENT DESCENT method!") if verbose else None
            if alpha is None and n_iter is not None:
                warnings.warn("Attempting to solve the linear regression using GRADIENT DESCENT but "
                              "the LEARNING RATE was NOT found! Proceed with the default value instead, "
                              "alpha = 0.001")
                alpha = 0.001  # Using the default learning rate
            elif alpha is not None and n_iter is None:
                warnings.warn("Attempting to solve the linear regression using GRADIENT DESCENT but "
                              "the NUMBER OF ITERATION was NOT found! Proceed with the default value instead, "
                              "n_iter = 1000")
                n_iter = 1000  # Using the default learning rate

        self.alpha = alpha  # Learning rate
        self.n_iter = int(n_iter)  # Number of iteration
        self.cost_function = cost  # Cost function between Mean Squared or Absolute Error
        self.verbose = verbose  # Verbosal option
        self.params = None

    def fit(self, X: np.array, y: np.array):
        """
        Fitting the Linear Regression model to the training dataset.

        Parameters
        ----------
        X:  ndarray with (n_train_samples, n_features) shape. Training data with,
            n_train_samples number of samples and n_features number of features.
        y:  ndarray with (n_train_samples,) shape. The training data label/target.

        Return
        ------
        self:   LinearRegression
                The fitted Linear Regression model, based on X and y train dataset.
        """
        self.n_samples, self.n_features = X.shape
        self.X_ = np.hstack([
            np.ones([self.n_samples, 1]),  # Placeholder for the intercepts
            feature_norm(X)  # Centering and normalize the data
        ])
        self.y_ = y[:, np.newaxis]

        # Initialize the training parameters
        self.params = np.zeros((self.n_features + 1, 1))

        if self.cost_function == 1:  # using the Least Absolute Deviation (LAD) Regression
            self._update_param_absolute()
        else:  # using the Ordinary Least Square (OLS) Regression
            self._update_param_squared()

        self.intercept_ = self.params[0]  # Bias
        self.coef_ = self.params[1:]  # Weights

        return self

    def _update_param_squared(self) -> None:
        """
        Gradient descent with Mean Squared Error cost function.
        Also known as Ordinary Least Squared (OLS) regression.

        Has 2 options to calculate the training parameters, such as:
            1. Using the iterative method with GRADIENT DESCENT.
            2. Direct calculation with NORMAL EQUATION.
        """
        if self._use_grad:  # Gradient Descent method
            for i in range(self.n_iter):
                grad = np.dot(self.X_.T, (np.dot(self.X_, self.params) - self.y_)) / self.n_samples
                self.params = self.params - self.alpha * grad  # Updating the parameters

        else:  # Normal equation method
            if self.n_features > 1e4:  # Avoiding large matrix operation!
                raise ValueError(f"The number of feature is TOO LARGE "
                                 f"(n_features = {self.n_features})! "
                                 f"Please use the GRADIENT DESCENT method instead "
                                 f"to avoid large matrix operation.")

            A = np.dot(self.X_.T, self.X_)
            if not check_invertible_array(A):
                warnings.warn("Singular matrix occurred! Please re-check the input features. "
                              "However, the program still proceed since it's using a pseudo-inverse calculation.")

            self.params = np.dot(np.dot(np.linalg.pinv(A), self.X_.T), self.y_)  # inv(A) dot X.T dot y

    def _update_param_absolute(self) -> None:
        """
        Gradient descent with Mean Absolute Error cost function.
        Also known as the Least Absolute Deviation (LAD) Regression model.

        Originally any cost function minimization can be done simply with the optimize module
        by scipy (scipy.optimize.minimize). However, since this model implementation only uses NumPy,
        another approach should be used to solve the LAD Regression model.
        """
        if self._use_grad:  # Gradient Descent method
            pass
        else:  # Normal equation method
            pass

    def predict(self, X) -> np.array:
        """
        Predict the target/label of the testing dataset,
        by performing the Linear Regression fitted by the training dataset.

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
        n_samples, _ = X.shape

        X_ = np.hstack([
            np.ones([n_samples, 1]),
            feature_norm(X)
        ])
        y_pred = np.dot(X_, self.params)
        return y_pred

    def get_params(self) -> Optional[np.array]:
        """
        Method to return the current parameters (weights/coefficients and biases/intercept).

        Return
        ------
        Params: None or ndarray with (n_features,) shape.
                The current parameters. Returns None if the object has NOT been FITTED before!
        """
        return self.params

    def score(self, X: Optional[np.array] = None, y: Optional[np.array] = None) -> Union[float, np.array]:
        """
        Calculate the R2 Score between the predicted target and the target labels.
        The maximum possible score is 1.0

        Parameters
        ----------
        X:  ndarray with (n_test_samples, n_features) shape. Testing data with,
            n_test_samples number of samples and n_features number of features.
            The target/label of this dataset will be predicted later.
        y:  ndarray with (n_test_samples,) shape.
            The testing data target/label.

        Return
        ------
        score:  float or ndarray with (n_outputs) if multi-outputs is predicted
                The R2 score between the target's prediction and label.
        """
        check_is_fitted(self)  # Check if self.fit is already performed before

        if X is None:
            X_ = self.X_
        else:
            n_samples, _ = X.shape
            X_ = np.hstack([
                np.ones([n_samples, 1]),
                feature_norm(X)
            ])

        if y is None:
            y_ = self.y_
        else:
            y_ = y[:, np.newaxis]

        y_pred = np.dot(X_, self.params)
        score = 1 - np.sum((y_ - y_pred) ** 2) / np.sum((y_ - np.mean(y_)) ** 2)

        return score
