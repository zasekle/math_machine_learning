# pytorch
# pytorch and tensorflow seem to be alternatives to each other
import matplotlib.pyplot as plt
import numpy as np


def intro_to_differential_calculus():
    print("intro_to_differential_calculus")

    # Calculus is the mathematical study of continuous change.
    # The derivative is the slope of the line.
    # Covered the basic stuff, velocity is the derivative of distance and acceleration is the derivative of velocity.


def intro_to_integral_calculus():
    print("intro_to_integral_calculus")

    # Integral calculus is the study of areas under curves. It facilitates the inverse of differential calculus.


def the_method_of_exhaustion():
    print("the_method_of_exhaustion")

    # The method of exhaustion will attempt to use smaller and smaller discrete values to eventually reach an infinitely
    #  small difference between them.


def calculus_of_infinitesimals():
    # Calculus of infinitesimals was a term coined by Leibniz that represents the method of exhaustion.

    # Clear the current figure and axes
    plt.clf()
    plt.cla()

    # Close all existing figures
    plt.close('all')

    # start, finish, n points
    x = np.linspace(-10, 10, 1000)

    y = x ** 2 + 2 * x + 2

    # While the below looks like a curve, it is actually a series of straight lines.
    fig, ax = plt.subplots()
    _ = ax.plot(x, y)

    # plt.show()

    # This will zoom into the curve very closely to show that it is a series of straight lines, not continuous.
    ax.set_xlim([-1.01, -0.99])
    ax.set_ylim([0.99, 1.01])
    _ = ax.plot(x, y)

    plt.show()


def limits_fn():
    intro_to_differential_calculus()
    intro_to_integral_calculus()
    the_method_of_exhaustion()
    # calculus_of_infinitesimals()
