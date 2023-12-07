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


def eigenvectors_and_eigenvalues_fn():
    segment_intro()
    applying_matrices()
    affine_transformations()
