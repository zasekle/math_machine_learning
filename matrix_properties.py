import numpy as np

# pytorch
import torch
# pytorch and tensorflow seem to be alternatives to each other
import tensorflow as tf

import matplotlib.pyplot as plt


def the_frobenius_norm():
    # the formula for this is ||x||f = sqrt(sum(x^2)) for all elements x in the matrix
    #   this is analogous to L2 norm of a vector
    #   measures the size of matrix in terms of euclidean distance
    #   it is the sum of the magnitude of all the VECTORS in the matrix

    x = np.array([[1, 2], [3, 4]])

    print(x)
    print(np.linalg.norm(x))


def matrix_multiplication():
    # in order to perform matrix multiplication, the number of rows in the LHS must be equal to the number of columns in
    #  RHS
    # matrix multiplication is not commutative, so AB != BA

    a = np.array([[3, 4], [5, 6], [7, 8]])
    b = np.array([[1], [2]])

    print(np.dot(a, b))

    b = np.array([[1, 9], [2, 0]])

    print(np.dot(a, b))


def symmetric_and_identity_matrices():
    # A symmetric matrix has the following properties
    #   Square
    #   The transpose must be equal to itself.

    x_sym = np.array([[0, 1, 2], [1, 7, 8], [2, 8, 9]])

    print(x_sym == x_sym.transpose())

    # An identity matrix is a special case of a symmetric matrix. It occurs when every element along the main diagonal
    # is 1 and all other elements are 0.
    #   The notation is In where n is the number or rows or columns.
    #   This matrix is special because an n-length vector is unchanged if multiplied by In.

    i = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    x_mul = np.dot(x_sym, i)
    print(x_mul == x_sym)


def matrix_multiplication_exercises():
    a = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    b = np.array([[-1], [1], [-2]])

    print(np.dot(a, b))

    a = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    b = np.array([[-1, 0], [1, 1], [-2, 2]])

    print(np.dot(a, b))


def matrix_inversion():
    # Matrix inversion is an alternative to manually solving with substitution or elimination.
    # Matrix inverse of X is denoted as X^-1.
    # Satisfies X^-1 * X = In
    # This can be used just like any inverse to solve an equation.
    # n = X*y
    # X^-1*n = y
    # However, it can only be used to solve an equation if a solution exists. Or in other words the matrix isn't
    #  singular (all columns must be non parallel in a geometric sense).

    x = np.array([[4, 2], [-5, -3]])

    x_inv = np.linalg.inv(x)

    print(x_inv)
    print(np.dot(x_inv, x))

    x = np.array([[-4, 1], [-8, 2]])

    print(x)

    # This matrix is NOT able to be inverted.
    # np.linalg.inv(x)


def matrix_properties_fn():
    the_frobenius_norm()
    matrix_multiplication()
    symmetric_and_identity_matrices()
    matrix_multiplication_exercises()
    matrix_inversion()
