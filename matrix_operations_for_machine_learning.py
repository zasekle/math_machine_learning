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


def data_compression_with_svd():
    from PIL import Image

    # Clear the current figure and axes
    plt.clf()
    plt.cla()

    img = Image.open('oboe-with-book.jpg')

    # Convert image to greyscale.
    img_gray = img.convert('LA')

    # Convert the image to a matrix that can be worked with in numpy.
    img_mat = np.array(list(img_gray.getdata(band=0)), float)
    img_mat.shape = (img_gray.size[1], img_gray.size[0])
    img_mat = np.matrix(img_mat)

    # Calculate SVD of the image.
    U, sigma, V = np.linalg.svd(img_mat)

    # Rebuilt the image from the matrices. This will be a compressed version (less compressed depending on the value of
    #  i). Note that this is stored by variance. So the leftmost values inside the 'sigma' matrix will be the most
    #  relevant. This allows for me to simply increase i below, including more columns and rows, and the image will get
    #  more defined.
    i = 64
    reconstructing = np.matrix(U[:, :i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :])

    _ = plt.imshow(reconstructing, cmap='gray')

    plt.show()


def the_moore_penrose_pseudo_inverse():
    # Remember that when calculating the inverse of a matrix there are quite a few conditions such as the matrix must
    #  be square. The moore-penrose pseudo inverse will take care of a lot of these problems.
    # For some matrix A, its pseudo inverse A^+ can be calculated by:
    #   A^+ = V * D^+ * U^T
    #   U, D and V are SVD of A
    #   D^+ = (D with reciprocal of all-non zero elements)^T

    a = np.array([[-1, 2], [3, -2], [5, 7]])

    u, d, v_t = np.linalg.svd(a)

    print(u)
    print(d)
    print(v_t)

    # Need to transform D into D^+
    d_diag = np.diag(d)
    d_inv = np.linalg.inv(d_diag)
    d_plus = np.concatenate((d_inv, np.array([[0], [0]])), axis=1)

    a_plus = np.dot(v_t.T, np.dot(d_plus, u.T))

    print(a_plus)

    # The built in method for calculating the pseudo inverse.
    print(np.linalg.pinv(a))


def matrix_operations_for_machine_learning_fn():
    singular_value_decomposition()
    # data_compression_with_svd()
    the_moore_penrose_pseudo_inverse()
