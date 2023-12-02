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


def matrix_properties_fn():
    the_frobenius_norm()
    matrix_multiplication()
