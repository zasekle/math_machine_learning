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

    # For a man cannot serve two masters. He will come to despise one and love the other. This means that I cannot love
    #  any other master besides God. So how do I put this in perspective with the other stuff? For example, does this
    #  mean that I can't love my mother? I suppose that it doesn't mean that I can't LOVE any source other than God,
    #  because it also says "Love your neighbor as yourself". It means I can't SERVE any other source. But does this
    #  mean that I can't do things in order to work? I have to work, AND I should do my job as if unto the Lord. So
    #  where does a Job fit in with this? Should I really be doing the bare minimum? That isn't doing my job as if unto
    #  the Lord. Should I spend all my time doing my job? Then I will less (?) time to spend with the Lord. No, I think
    #  a large part of my life is supposed to be toil. It is a curse from the garden if Eden. So, I guess it is ok to
    #  work, but I need to keep the Lord at #1 in my life. NO, I need to NOT serve my job while still having a job and
    #  doing it unto the Lord. So its because its all about the Lord. I serve him at all times, that means going and
    #  getting a job because he tells me to, doing an AMAZING job because he tells me to and sharing the gospel because
    #  he tells me to right? I think this is closer at least.


def regression_with_the_pseudo_inverse():
    # It is possible to do regression (the fitting to a line) using the pseudo inverse. Usually in real world data you
    #  have more equations that you do unknowns (an overdetermined system), a lot more. Or in deep learning you have
    #  more unknowns than you have equations (an underdetermined system). This means that it is likely that no solution
    #  for the system exists.

    x1 = [0, 1, 2, 3, 4, 5, 6, 7.]  # E.g.: Dosage of drug for treating Alzheimer's disease
    y = [1.86, 1.31, .62, .33, .09, -.67, -1.23, -1.37]  # E.g.: Patient's "forgetfulness score"

    x0 = np.ones(8)

    # Essentially what is happening here is that any equation of the form y = mx OR in vector form y = mX must go
    # through the origin. However, we don't want that limitation on our regression. So in order to do this we add the
    # variable for the y-intercept. So we have y = mx + b OR in vector form you combine the unknowns into the vector and
    # extract the constants in front of them so for example [[1, 2], [1, 3], [1, 4]] * [b, m]. This means that a column
    # of ones must be added to represent the new variable b.
    X = np.concatenate((np.matrix(x0).T, np.matrix(x1).T), axis=1)

    w = np.dot(np.linalg.pinv(X), y)

    print(w)

    # Clear the current figure and axes
    plt.clf()
    plt.cla()

    b = np.asarray(w).reshape(-1)[0]
    m = np.asarray(w).reshape(-1)[1]

    title = 'Clinical Trial'
    x_label = 'Drug dosage (mL)'
    y_label = 'Forgetfulness'
    fig, ax = plt.subplots()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    ax.scatter(x1, y)

    x_min, x_max = ax.get_xlim()
    y_at_x_min = m * x_min + b
    y_at_x_max = m * x_max + b

    ax.set_xlim([x_min, x_max])
    _ = ax.plot([x_min, x_max], [y_at_x_min, y_at_x_max], c='C01')

    plt.show()


def matrix_operations_for_machine_learning_fn():
    singular_value_decomposition()
    # data_compression_with_svd()
    the_moore_penrose_pseudo_inverse()
    regression_with_the_pseudo_inverse()
