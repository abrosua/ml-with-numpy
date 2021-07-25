from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

from models.linear import LinearRegression
from models.pca import PCA
from models.knn import KNeighborsRegressor


def print_results(y_test, y_pred):
    # Print the Mean Squared Error
    print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
    # Print the coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))


if __name__ == "__main__":
    # Load the diabetes dataset
    X, y = datasets.load_boston(return_X_y=True)
    print(f"Total {len(y)} samples are found!")

    # Use PCA to reduce the feature
    num_pca = 3  # 3 Components are already enough to cover more than 99% of the data variance
    print(f"Performing PCA with {num_pca} component(s)")
    pca = PCA(num_pca)
    X_pca = pca.fit_transform(X)
    print(f"Cumulative explained variance ratio: {round(100 * pca.explained_variance_ratio_.sum(), 2)}%")

    # Splitting dataset into train and test set (80:20)
    order = np.random.permutation(len(y))
    test_portion = 0.2
    point_divide = int(test_portion * len(y))
    X_train = X_pca[order[point_divide:]]
    X_test = X_pca[order[:point_divide]]
    y_train = y[order[point_divide:]]
    y_test = y[order[:point_divide]]

    ## ------------------- Performing Regression! -------------------

    # Solve using kNN Regression
    print("\n---------- k-NN Regression ----------")
    n_neighbors = 10
    knn_np = KNeighborsRegressor(k=n_neighbors)  # Instantiate the k-NN model
    knn_np.fit(X_train, y_train)  # Fit the k-NN model to the training set!
    y_pred_knn = knn_np.predict(X_test)  # Make predictions using the testing set
    print_results(y_test, y_pred_knn)

    # Solve using OLS Regression (Normal Equation)
    print("\n---------- OLS Regression with Normal Equation ----------")
    ols_norm = LinearRegression(normalize=False, cost=2)  # Mean Squared Error (MSE) as the cost function
    ols_norm.fit(X_train, y_train)  # Fit the OLS Regression model to the training set!
    ols_norm_pred = ols_norm.predict(X_test)  # Make predictions using the testing set
    print_results(y_test, ols_norm_pred)

    # Solve using OLS Regression (Gradient Descent)
    print("\n---------- OLS Regression with Gradient Descent ----------")
    ols_grad = LinearRegression(alpha=0.01, n_iter=1000, normalize=True, cost=2)  # MSE as the cost function
    ols_grad.fit(X_train, y_train)  # Fit the OLS Regression model to the training set!
    ols_grad_pred = ols_grad.predict(X_test)  # Make predictions using the testing set
    print_results(y_test, ols_grad_pred)

