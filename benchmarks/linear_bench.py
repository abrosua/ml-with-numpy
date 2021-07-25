import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

from models.linear import LinearRegression


def print_results(fitted_model, y_test, y_pred):
    print('Coefficients: ', fitted_model.coef_)
    print('Intercept: ', fitted_model.intercept_)
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(y_test, y_pred))


if __name__ == "__main__":
    ############## You May Use dataset and the test/training split from sklearn ################
    # Load the diabetes dataset
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

    # Use only one feature
    diabetes_X = diabetes_X[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes_y[:-20]
    diabetes_y_test = diabetes_y[-20:]

    ############## You May Use dataset and the test/training split from sklearn ################
    print("Comparing the sklearn results and the developed NumPy solutions!")

    ## Sklearn OLS Regression
    print('\n------- OLS Regression with Scikit-learn -------')
    regr = linear_model.LinearRegression()  # Create linear regression object
    regr.fit(diabetes_X_train, diabetes_y_train)  # Train the model using the training sets
    diabetes_y_pred = regr.predict(diabetes_X_test)  # Make predictions using the testing set
    print_results(regr, diabetes_y_test, diabetes_y_pred)  # Print the results and coefficients

    ## NumPy with NORMAL EQUATION
    print('\n------- OLS (NORMAL EQUATION) Regression with NumPy -------')
    ols_norm = LinearRegression(normalize=False, cost=2)  # Mean Squared Error (MSE) as the cost function
    ols_norm.fit(diabetes_X_train, diabetes_y_train)  # Train the model using the training sets
    ols_norm_pred = ols_norm.predict(diabetes_X_test)  # Make predictions using the testing set
    print_results(ols_norm, diabetes_y_test, ols_norm_pred)  # Print the results and coefficients

    ## NumPy with GRADIANT DESCENT
    print('\n------- OLS (GRADIENT DESCENT) Regression with NumPy -------')
    ols_grad = LinearRegression(alpha=0.01, n_iter=1000, normalize=True, cost=2)  # MSE as the cost function
    ols_grad.fit(diabetes_X_train, diabetes_y_train)  # Train the model using the training sets
    ols_grad_pred = ols_grad.predict(diabetes_X_test)  # Make predictions using the testing set
    print_results(ols_grad, diabetes_y_test, ols_grad_pred)  # Print the results and coefficients

    ## NumPy with LAD
    print('\n------- LAD Regression with NumPy -------')
    lad_np = LinearRegression(n_iter=200, normalize=False, cost=1)  # Mean Absolute Error (MAE) as the cost function
    lad_np.fit(diabetes_X_train, diabetes_y_train)  # Train the model using the training sets
    lad_np_pred = lad_np.predict(diabetes_X_test)  # Make predictions using the testing set
    print_results(lad_np, diabetes_y_test, lad_np_pred)  # Print the results and coefficients

    # ------------ Plot outputs ------------
    plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
    plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3,
             label='Sklearn OLS')
    plt.plot(diabetes_X_test, ols_norm_pred, color='red', linewidth=2, linestyle='dashed',
             label='Numpy OLS with normal eq.')
    plt.plot(diabetes_X_test, ols_grad_pred, color='green', linewidth=2, linestyle='dotted',
             label='Numpy OLS with gradient descent')
    #plt.plot(diabetes_X_test, lad_np_pred, color='orange', linewidth=2, linestyle='--',
    #         label='Numpy LAD')

    plt.legend()
    plt.title("Linear Regression with sklearn and NumPy implementations")
    plt.show()
