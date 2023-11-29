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


def data_structures_for_linear_algebra_fn():
    what_is_linear_algebra()
    # plotting_a_system_of_linear_equations()
    # linear_algebra_exercise()
    tensors()
    scalars()
    vectors()
    norms_and_unit_vectors()
