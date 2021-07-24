import numpy as np
from sklearn.decomposition import PCA as PCA_sk

from models.pca import PCA as PCA_np


if __name__ == "__main__":
    num_components = 2
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype="float")
    print(A)

    # create the PCA instance
    print("\n\nPCA with sklearn..............")
    pca_sk = PCA_sk(n_components=num_components)
    pca_sk.fit(A)
    # access values and vectors
    print("Components:")
    print(pca_sk.components_)
    print("Explained variance:")
    print(pca_sk.explained_variance_)
    # transform data
    A_sk = pca_sk.transform(A)
    print("PCA results with sklearn:")
    print(A_sk)

    # PCA with numpy
    print("\n\nPCA with NumPy..............")
    pca_np = PCA_np(n_components=num_components)
    pca_np.fit(A)
    # access values and vectors
    print("Components:")
    print(pca_np.components_)
    print("Explained variance:")
    print(pca_np.explained_variance_)
    # transform data
    A_np = pca_np.transform(A)
    print("PCA results with sklearn:")
    print(A_np)

    # Difference
    diff = (np.abs(A_np) - np.abs(A_sk))
    print(f"\n\nResult difference:")
    print(diff)

    # asserting the components
    assert np.allclose(abs((pca_sk.components_ * pca_np.components_).sum(1)), 1)
