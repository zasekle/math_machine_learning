# pytorch
# pytorch and tensorflow seem to be alternatives to each other
import matplotlib.pyplot as plt
import numpy as np


def singular_value_decomposition():
    # SVD (Singular Value Decomposition) is similar to eigendecomposition.
    #   This is applicable to any real-valued matrix, not just square matrices.
    #   It decomposes the matrix into singular vectors and singular values (just like eigenvectors).
    #   For some matrix A, A = UDV^T. U has the vectors, D is a diagonal matrix and V^T is a square matrix with the
    #    number of columns of A.
    #   Left-singular vectors of A = eigenvectors of A*A^T
    #   Right-singular vectors of A = eigenvectors of A^T*A
    #   Non-zero singular values of A = square roots of eigenvectors of A*A^T = square root of eigenvectors of A^T*A

    a = np.array([[-1, 2], [3, -2], [5, 7]])

    # v is already transposed at this point
    u, d, v_t = np.linalg.svd(a)

    print(u)
    print(d)
    print(v_t)

    d_diag = np.diag(d)

    # must concatenate on a row in order to allow the matrix multiplication between d_diag and v_t to work
    d_fin = np.concatenate((d_diag, [[0, 0]]), axis=0)

    print(np.dot(u, np.dot(d_fin, v_t)))


def matrix_operations_for_machine_learning_fn():
    singular_value_decomposition()
