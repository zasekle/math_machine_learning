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


def diagonal_matrices():
    # A diagonal matrix has nonzero elements along the main diagonal and zeros everywhere else. The identify matrix is
    #  an example of a diagonal matrix. If the matrix is square, it can be denoted as diag(x) where x is the vector of
    #  the main-diagonal elements.
    #    Multiplication: diag(x)y = x ⨀ y
    #    Inversion: diag(x)-1 = diag[1/x1, ..., 1/xn]T; Note that we cannot have zero here because zero cannot be the
    #     denominator
    #    If the diagonal matrix is non-square, this is still computationally efficient.
    print("diagonal_matrices")


def orthogonal_matrices():
    # Or orthogonal matrix, orthonormal vectors make up all rows AND all columns.
    #   This means AT * A = A * AT = I;
    #     If we multiply the above by A-1 we end up with the below equation.
    #   AT = A-1 * I = A-1
    #     This means that because calculation AT is cheap, therefore calculating A-1
    print("orthogonal_matrices")


def check_if_orthogonal_matrix(a):
    # all rows are orthogonal
    b = np.dot(a[0], a[1])
    if b != 0:
        return False

    b = np.dot(a[0], a[2])
    if b != 0:
        return False

    b = np.dot(a[1], a[2])
    if b != 0:
        return False

    # all columns are orthogonal
    b = np.dot(a[:, 0], a[:, 1])
    if b != 0:
        return False

    b = np.dot(a[:, 0], a[:, 2])
    if b != 0:
        return False

    b = np.dot(a[:, 1], a[:, 2])
    if b != 0:
        return False

    # all rows are normal
    b = np.linalg.norm(a[0])
    if b != 1.0:
        return False

    b = np.linalg.norm(a[1])
    if b != 1.0:
        return False

    b = np.linalg.norm(a[2])
    if b != 1.0:
        return False

    # all columns are normal
    b = np.linalg.norm(a[:, 0])
    if b != 1.0:
        return False

    b = np.linalg.norm(a[:, 1])
    if b != 1.0:
        return False

    b = np.linalg.norm(a[:, 2])
    if b != 1.0:
        return False

    return True


def orthogonal_matrix_exercises():
    a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    print(check_if_orthogonal_matrix(a))

    a = np.array([[2 / 3, 1 / 3, 2 / 3], [-2 / 3, 2 / 3, 1 / 3], [1 / 3, 2 / 3, -2 / 3]])

    print(check_if_orthogonal_matrix(a))


def matrix_properties_fn():
    the_frobenius_norm()
    matrix_multiplication()
    symmetric_and_identity_matrices()
    matrix_multiplication_exercises()
    matrix_inversion()
    diagonal_matrices()
    orthogonal_matrices()
    orthogonal_matrix_exercises()
