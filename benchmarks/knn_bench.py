# #############################################################################
# Generate sample data
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

from models import knn


if __name__ == "__main__":
    np.random.seed(0)
    X = np.sort(5 * np.random.rand(40, 1), axis=0)
    T = np.linspace(0, 5, 500)[:, np.newaxis]
    y = np.sin(X).ravel()

    # Add noise to targets
    y[::5] += 1 * (0.5 - np.random.rand(8))

    # #############################################################################
    # Fit regression model
    n_neighbors = 10

    # ------ KNN with scikit-learn ------
    knn_sk = neighbors.KNeighborsRegressor(n_neighbors)
    model_sk = knn_sk.fit(X, y)
    y_sk = model_sk.predict(T)

    # ------ KNN with NumPy ------
    knn_np = knn.KNeighborsRegressor(k=n_neighbors)
    model_np = knn_np.fit(X, y)
    y_np = model_np.predict(T)

    # ------ COMPARISON ------
    y_comp = np.dstack([y_sk, y_np])
    RMSE = np.sqrt(np.mean((y_sk - y_np) ** 2))
    print(f"RMSE between sklearn and numpy model: {RMSE}")

    # ------ Plotting ------
    plt.scatter(X, y, color='blue', label='data')
    plt.plot(T, y_sk, color='green', label='sklearn prediction')
    plt.plot(T, y_np, color='red', linestyle='--', label='NumPy prediction')

    plt.legend()
    plt.title("KNeighborsRegressor")

    plt.show()
