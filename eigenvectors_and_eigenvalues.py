# pytorch
# pytorch and tensorflow seem to be alternatives to each other
import matplotlib.pyplot as plt
import numpy as np


def segment_intro():
    print("segment_intro")


def applying_matrices():
    u = np.array([[2], [5], [-3]])
    u2 = np.array([[0], [-4], [6]])
    b = np.array([[2, 0, -1], [-2, 3, 1], [0, 4, -1]])

    # create an identity matrix of size 3
    i3 = np.eye(3)

    print(np.dot(i3, u))

    u_b = np.dot(b, u)
    print(np.dot(b, u))

    u2_b = np.dot(b, u2)

    # combines the single column vectors into a two column matrix
    u_c = np.column_stack((u_b, u2_b))

    print(u_c)


def plot_vectors(vectors, colors):
    """
    Plot one or more vectors in a 2D plane, specifying a color for each.

    Arguments
    ---------
    vectors: list of lists or of arrays
        Coordinates of the vectors to plot. For example, [[1, 3], [2, 2]]
        contains two vectors to plot, [1, 3] and [2, 2].
    colors: list
        Colors of the vectors. For instance: ['red', 'blue'] will display the
        first vector in red and the second in blue.

    Example
    -------
    plot_vectors([[1, 3], [2, 2]], ['red', 'blue'])
    plt.xlim(-1, 4)
    plt.ylim(-1, 4)
    """
    plt.figure()
    plt.axvline(x=0, color='lightgray')
    plt.axhline(y=0, color='lightgray')

    for i in range(len(vectors)):
        x = np.concatenate([[0, 0], vectors[i]])
        plt.quiver([x[0]], [x[1]], [x[2]], [x[3]],
                   angles='xy', scale_units='xy', scale=1, color=colors[i], )


def affine_transformations():
    # affine transformations are transformations that changes the geometry between vectors, but preserves parallelism
    #  (this means if they are parallel before, they will be parallel after) between them. Some common affine
    #  transformations include reflection, scaling, shearing (this displaces each point in a fixed direction) and
    #  rotation.

    v = np.array([3, 1])

    # this transformation flips the vector over the x-axis (reflection)
    t = np.array([[1, 0], [0, -1]])

    plot_vectors([v, np.dot(t, v)], ['lightblue', 'blue'])

    plt.xlim(-1, 5)
    _ = plt.ylim(-1, 5)
    plt.show()


def eigenvectors_and_eigenvalues():
    # An eigenvector of a linear transformation (represented by a matrix) is a non-zero vector that changes at most by
    #  a scalar factor when that linear transformation is applied to it.
    # An eigenvalue is the scalar by which the eigenvector is scaled during the transformation. Note that this is the
    #  amount that the vector changed AFTER the change not the scalar or matrix that caused the change.
    # An eigenvector stays along the same axis in the sense that it doesn't rotate or change direction under the
    #  transformation.

    # To conceptualize this, think of a as the transformation matrix for a 2d vector. Of all the vectors that exist
    #  inside 2d space, there are a few where this transformation will only scale them. These vectors are called
    #  eigenvectors and the amount they are scaled is called the eigenvalue.
    a = np.array([[-1, 4], [2, -2]])

    # This will calculate possible eigenvalues and eigenvectors for the given array. The column V[:,i] is the
    #  eigenvector corresponding to the eigenvalue lambdas[i].
    values, vectors = np.linalg.eig(a)

    print(values)
    print(vectors)

    v = vectors[:, 0]
    Av = np.dot(a, v)

    # The scalar times the eigenvector and the original times the eigenvector will give the same result. That is
    #  because of the relationship of Av = Lv where v is the eigenvector, L is the eigenvalue and A is the original
    #  array.
    print(values[0] * v == Av)

    plot_vectors([Av, v], ['blue', 'lightblue'])
    plt.xlim(-1, 2)
    _ = plt.ylim(-1, 2)
    # plt.show()


def matrix_determinants():
    # A matrix determinant maps a square matrix to a scalar.
    #   It allows us to determine whether matrix can be inverted. If the det(X) = 0, then the matrix has no inversion.

    x = np.array([[4, 2], [-5, -3]])

    print(np.linalg.det(x))


def determinants_of_larger_matrices():
    # Calculating the determinant of matrices larger than 2x2 is recursive in nature. Essentially you break the matrix
    #  down into smaller sections and calculate the value of those, then add or subtract them together.

    x = np.array([[1, 2, 4], [2, -1, 3], [0, 5, 1]])

    print(np.linalg.det(x))


def determinant_exercises():
    x = np.array([[25, 2], [3, 4]])

    # 94
    print(np.linalg.det(x))

    x = np.array([[-2, 0], [0, -2]])

    # 4
    print(np.linalg.det(x))

    x = np.array([
        [2, 1, -3],
        [4, -5, 2],
        [0, -1, 3]
    ])

    # 2 * (-15 + 2) = -26
    # 1 * (12 - 0) = 12
    # -3 * (-4 - 0) = 12
    # -26 - 12 + 12 = -26
    print(np.linalg.det(x))


def determinants_and_eigenvalues():
    # The relationship between determinants and eigenvalues is `det(X) = product of all eigenvalues of X`.
    # abs(det(x)) quantifies volume change as a result of applying x.
    #   If det(x) = 0, then x collapses space completely in at least one dimension, eliminating all volume.
    #   If 0 abs(det(x)) < 1, then x contracts volume to some extent.
    #   If abs(det(x)) == 1, then is will preserve volume.
    #   If abs(det(x)) > 1, then the volume will increase.

    x = np.array([
        [1, 2, 4],
        [2, -1, 3],
        [0, 5, 1],
    ])

    lambdas, V = np.linalg.eig(x)
    det = np.linalg.det(x)

    # These are the same because of the relationship between the product of the eigenvalues and teh determinant.
    print(np.product(lambdas))
    print(det)

    # This value is ~20, so the volume will increase with this transformation.
    print(np.abs(det))


def eigendecomposition():
    # A = V Λ V^-1
    #   A is the original square matrix that we are decomposing.
    #   V contains all the eigenvectors of A as its columns.
    #   Λ (capital lambda) is a diagonal matrix containing the eigenvalues of A.
    #   V^-1 is the inverse of matrix V.
    # The decomposition of a matrix into eigenvectors and eigenvalues reveals characteristics of the matrix, e.g.
    #   Matrix is singular if and only if any of its eigenvalues are zero.
    #   Under specific conditions, we ca n optimize quadratic expressions
    #     Maximum of f(x) = largest eigenvalue
    #     Minimum of f(x) = smallest eigenvalue.

    a = np.array([
        [4, 2],
        [-5, -3]
    ])

    lambdas, v = np.linalg.eig(a)

    v_inv = np.linalg.inv(v)

    diag = np.diag(lambdas)

    print(v)
    print(v_inv)
    print(diag)

    # A = V Λ V^-1
    print(a)
    print(np.dot(v, np.dot(diag, v_inv)))

    # eigendecomposition is not possible with all matrices. And in some cases where it is possible, the
    #  eigendecomposition involves complex numbers instead of straightforward real numbers.
    # In machine learning, however, we are typically working with real symmetric matrices, which can be conveniently and
    #  effectively decompose into real-only eigenvectors and real-only eigenvalues.

    # A = Q Λ Qt
    #   This formula is the same as the above formula except V is not Q and it is the transpose of Q instead of the
    #   inverse of V.
    #   This formula is important in machine learning because the transpose is cheaper to calculate that the inverse.

    a = np.array([[2, 1], [1, 2]])

    lambdas, q = np.linalg.eig(a)

    diag = np.diag(lambdas)

    print(a)
    print(np.dot(q, np.dot(diag, q.T)))


def eigenvectors_and_eigenvalues_fn():
    segment_intro()
    applying_matrices()
    # affine_transformations()
    # eigenvectors_and_eigenvalues()
    matrix_determinants()
    determinants_of_larger_matrices()
    determinant_exercises()
    determinants_and_eigenvalues()
    eigendecomposition()
