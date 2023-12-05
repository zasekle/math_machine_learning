import numpy as np

# pytorch
import torch
# pytorch and tensorflow seem to be alternatives to each other
import tensorflow as tf

import matplotlib.pyplot as plt


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
    #   are zero

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


def tensor_operations_fn():
    tensor_transposition()
    basic_tensor_arithmetic()
    tensor_reduction()
    the_dot_product()
    exercises_on_tensor_operations()
    solving_linear_systems_with_substitution()
    solving_linear_systems_with_elimination()
    # visualizing_linear_systems()
