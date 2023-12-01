import numpy as np

# pytorch
import torch
# pytorch and tensorflow seem to be alternatives to each other
import tensorflow as tf

import matplotlib.pyplot as plt


def what_is_linear_algebra():
    print("what_is_linear_algebra() running!")

    # Algebra is arithmetic that include non-numerical entities like x. For example, `2x + 5 = 25`. Linear algebra
    #  is algebra is only used on linear equations. A more formal definition of linear algebra is "Solving for
    #  unknowns within a system of linear equations".

    # xf - xi = v*t
    # Robber, xi = 0
    # x = 150 * t
    # x/150 = t
    # Cop, xi = -(180 * 5/60)
    # x + 15 = 180 * t
    # x/180 + 1/12 = t
    # x/150 = x/180 + 1/12
    # x/150 - x/180 = 1/12
    # (6x - 5x)/900 = 1/12
    # x/900 = 1/12
    # x = 75 km


def plotting_a_system_of_linear_equations():
    t = np.linspace(0, 40, 1000)  # start, finish, n points

    d_r = 2.5 * t
    d_s = 3 * (t - 5)

    fig, ax = plt.subplots()
    plt.title('A Bank Robber Caught')
    plt.xlabel('time (in minutes)')
    plt.ylabel('distance (in km)')
    ax.set_xlim([0, 40])
    ax.set_ylim([0, 100])
    ax.plot(t, d_r, c='green')
    ax.plot(t, d_s, c='brown')
    plt.axvline(x=30, color='purple', linestyle='--')
    _ = plt.axhline(y=75, color='purple', linestyle='--')

    plt.show()


def linear_algebra_exercise():
    t = np.linspace(0, 50, 1000)  # start, finish, n points

    d_a = t
    d_b = 4 * (t - 30)

    fig, ax = plt.subplots()
    plt.title('Solar panels')
    plt.xlabel('time (in days)')
    plt.ylabel('power (in kJ)')
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 70])
    ax.plot(t, d_a, c='green')
    ax.plot(t, d_b, c='brown')
    plt.axvline(x=40, color='purple', linestyle='--')
    _ = plt.axhline(y=40, color='purple', linestyle='--')

    plt.show()


def tensors():
    print("tensors")

    # A scalar is a single value.
    # A vector is a one dimensional array.
    # A matrix is essentially a two dimensional array.
    # A tensor generalizes vectors and matrices to any number of dimensions.


def scalars():
    # scalar properties
    # no dimensions
    # single number
    # denoted in lowercase, italics
    # should be typed, like all other tensors: e.g. int, float32

    x = 25

    print(x)
    print(type(x))

    y = 3.5

    print(type(y))

    x_pt = torch.tensor(25)

    print(x_pt)
    print(x_pt.shape)

    x_tf = tf.Variable(25, dtype=tf.int16)

    print(x_tf)
    print(x_tf.shape)

    y_tf = tf.Variable(3, dtype=tf.int16)

    print(x_tf + y_tf)

    tf_sum = tf.add(x_tf, y_tf)

    print(tf_sum)

    print(tf_sum.numpy())

    print(type(tf_sum.numpy()))

    tf_float = tf.Variable(25., dtype=tf.float16)

    print(tf_float)


def vectors():
    # vector properties
    # one-dimensional array of numbers
    # denoted in lowercase, italics, bold
    # arranged in an order, so element can be accessed by its index
    #    Elements are scalars so not bold
    # can represents a point in space (2 elements is 2D, etc)

    # transition
    # this will transition a row vector to a column vector or vise versa
    # [x1, x2]T = [x1,
    #              x2]
    # shape is the vectors (columns, rows) so the above vectors are (1, 2) and (2, 1) respectively

    # type argument is optional, e.g.: dtype=np.float16
    x = np.array([25, 2, 5])

    print(len(x))
    print(x.shape)
    print(type(x))
    print(x[0])
    print(type(x[0]))

    # notice the double brackets, this will give it a shape and allow transposition
    y = np.array([[4, 5, 6]])
    y_t = y.transpose()

    print(y)
    print(y.shape)
    print(y_t)
    print(y_t.shape)

    # transpose brings it back to the original
    print(y_t.transpose())
    print(y_t.transpose().shape)

    # creating an array in pytorch
    x_pt = torch.tensor([1, 2, 3])

    print(x_pt)

    # creating an array in tensorflow
    x_tf = tf.Variable([6, 5, 4])

    print(x_tf)


def norms_and_unit_vectors():
    # vectors can be used to represent a magnitude and direction from the origin (a line segment)
    # norms are functions that quantify vector magnitude
    # the most straightforward norm is the distance formula called L2 norm expanded for multiple dimensions
    #   sqrt(a^2+b^2...)= ||x||2
    #   this measures the simple Euclidean distance from the origin
    #   sometimes simple called (||x||)

    x = np.array([1, 2, 3])

    # L2 norm
    print(np.linalg.norm(x))

    # a unit vector is a vector where the magnitude is 1

    # L1 norm
    # this is the sum of absolute values of vector elements
    #   sum(abs(xi)) = ||x||1

    # squared L2 norm
    # this is L2 norm squared (can simply remove the sqrt before the calculation is done)
    # cheap operation comparatively to L2 (no sqrt)
    # grows slowly near origin when using very small numbers
    print(np.dot(x, x))

    # max norm (or L∞ norm)
    # this returns the largest element of the absolute value of the elements (so for [1,2,3] it would return 3)
    # max(|xi|)
    print(np.max([np.abs(x[0]), np.abs(x[1]), np.abs(x[2])]))

    # generalized case of norms is a specific equation called Lp norm
    # (sum(abs(xi)^p))^1/2 = ||x||p
    # p must be a real number >= 1
    # this is the reason they are known by numbers, L2 norm means p = 2


def basis_orthogonal_and_orthonormal_vectors():
    # basis vectors are a set of vectors that can be scaled to represent any vector in a given vector space
    #   for example set{(1,0),(0,1)} are the basis vectors for 2D space

    # orthogonal vectors are at a 90 degree angle to each other, this means they are 90 degrees from each other, and the
    # definition means that only 2 can exist at a time, there are infinite configurations
    #   the dot product is equal to zero on orthogonal vectors

    # orthonormal vectors are orthogonal and all have the unit norm

    i = np.array([1, 0])
    j = np.array([0, 1])

    print(np.dot(i, j))


def matrix_tensors():
    # a matrix is a two-dimensional array of numbers
    #  denoted in uppercase, italics, bold
    #  height given priority ahead of width in notation, i.e.: (nrow, ncol)
    #    note that this is because generic notation is (i, j, k, l, ...) and b/c i is first, rows come first
    #  individual scalar elements denoted in uppercase, italics only
    #  colon represents an entire row or column, so x:,1 is the first column

    x = np.array([[25, 2], [5, 26], [3, 7]])

    print(x)
    print(x.shape)
    print(x.size)

    # slicing
    print(x[:, 0])
    print(x[1, :])
    print(x[0:2, 0:2])

    # pytorch
    x_pt = torch.tensor([[25, 2], [5, 26], [3, 7]])
    print(x_pt)

    # tensorflow
    x_tf = tf.Variable([[25, 2], [5, 26], [3, 7]])
    print(x_tf)

    # the number of dimensions, so 2 in this case
    print(tf.rank(x_tf))


def generic_tensor_notation():
    # generic tensors
    # upper-case, bold, italics, sans serif
    # in a 4-tensor x, element at (i, j, k, l) is represented x(i, j, k, l)

    images_pt = torch.zeros([32, 28, 28, 3])

    print(images_pt)


def exercises_on_algebra_data_structures():
    # transpose of vector

    q_1 = np.array([[25], [2], [-3], [-23]])

    print(q_1.transpose())

    # dimensions of vector

    q_2 = np.array([[42, 4, 7, 99], [-99, -3, 17, 22]])

    print(q_2.shape)
    # mathematical notation is 2,3 because it is not zero-indexed
    print(q_2[1, 2])


def tensor_transposition():
    # transpose of scalar is itself
    # transpose of vector, convert column to row
    # transpose of a matrix is flipping over the rows and axis, so x(i,j) becomes x(j,i)

    x = np.array([[25, 2], [5, 26], [3, 7]])

    # transposition
    print(x.T)


def basic_tensor_arithmetic():
    x = np.array([[25, 2], [5, 26], [3, 7]])

    # a scalar is the simplest value, the operation is simply done to each element
    print(x * 2)
    print(x + 2)
    # standard order of operations apply
    print(x * 2 + 2)

    a = x + 2
    print(a)

    # note that matrices of the same size have a special exception when dealing with operations, the operations are
    #  applied element wise instead of the standard rules for matrix multiplication
    print(x + a)
    print(x * a)


def tensor_reduction():
    # we can reduce vectors, a simple way is the sum function

    x = np.array([[25, 2], [5, 26], [3, 7]])

    print(x.sum())

    # reduction can also be done with reduction along all or a selection of axes, e.g.
    #   maximum
    #   minimum
    #   mean
    #   product


def the_dot_product():
    # this is a way of multiplying two vectors of equal length to get a scalar
    #   the equation between vectors x and y is sum(xi,yi)
    # the geometric interpretation is
    #   a dot b = |a| * |b| * cos(angle)
    # conceptually the dot produce represents projection and similarity, so how much one vector projects onto another,
    #  or another way of looking at it is how similar the vectors are, a dot product where the vectors are 90 degrees
    #  are zero

    # as a side note, the cross product result represents a vector that is perpendicular to both vectors

    x = np.array([2, 3, 4])
    y = np.array([8, 9, 10])

    print(np.dot(x, y))


def exercises_on_tensor_operations():
    y = np.array([[42, 4, 7, 99], [-99, -3, 17, 22]])

    print(y.transpose())

    a = np.array([[25, 10], [-2, 1]])
    b = np.array([[-1, 7], [10, 8]])

    print(a * b)

    i = np.array([-1, 2, -3])
    j = np.array([5, 10, 0])

    print(np.dot(i, j))


def solving_linear_systems_with_substitution():
    print("solving_learn_systems()")

    # this is what I consider to be the goto method of solving linear equations, it means you substitute variables
    #   x = 3y
    #   x + 4y = 7
    #   therefore
    #   3y + 4y = 7

    # x + y = 6 and 2x + 3y = 16
    # 2(6 - y) + 3y = 16
    # 12 + y = 16
    # y = 4

    # y = 4x + 1 and -4x + y = 2
    # -4x + 4x + 1 = 2
    # 1 = 2
    # no solution (parallel lines)


def solving_linear_systems_with_elimination():
    print("solving_linear_systems_with_elimination()")

    # this means eliminating a variable be running an operation between the equations
    #   2x - 3y = 15 and 4x + 10y = 14
    #   4x - 6y = 30
    #   4x + 10y = 14
    #   subtract them
    #   -16y = 16
    #   y = -1
    #   x = 6


def visualizing_linear_systems():
    # in a simple system the intersection is the solution to the system, e.g.
    # y = 3x
    # y = 1 + 5x/2

    x = np.linspace(-10, 10, 1000)  # start, finish, n points

    y1 = 3 * x
    y2 = 1 + 5 * x / 2

    fig, ax = plt.subplots()
    plt.title('Visualizing')
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_xlim([0, 3])
    ax.set_ylim([0, 8])
    ax.plot(x, y1, c='green')
    ax.plot(x, y2, c='brown')
    plt.axvline(x=2, color='purple', linestyle='--')
    _ = plt.axhline(y=6, color='purple', linestyle='--')

    plt.show()


def data_structures_for_linear_algebra_fn():
    what_is_linear_algebra()
    # plotting_a_system_of_linear_equations()
    # linear_algebra_exercise()
    tensors()
    scalars()
    vectors()
    norms_and_unit_vectors()
    basis_orthogonal_and_orthonormal_vectors()
    matrix_tensors()
    generic_tensor_notation()
    exercises_on_algebra_data_structures()
    tensor_transposition()
    basic_tensor_arithmetic()
    tensor_reduction()
    the_dot_product()
    exercises_on_tensor_operations()
    solving_linear_systems_with_substitution()
    solving_linear_systems_with_elimination()
    # visualizing_linear_systems()
