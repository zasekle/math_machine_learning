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


def calculus_applications():
    print("calculus_applications")

    # Derivatives can be used to show the minimum or maximum of things. These things can be cost OR used in machine
    #  learning to find the minimum of a curve (I assume where the formula is not solvable).
    #  2nd and higher derivatives can be taken more quickly and so they can be used in optimizations.

    # Integrals can find the area under a curve. This allows for something called "Receiver operating characteristic" to
    #  be used. This is a useful algorithm for binary situations such as "is this a dog" or "is it true/false" things
    #  along those lines.


def calculating_limits():
    print("calculating_limits")

    # For continuous functions, limits are trivial to calculate. For example, as x -> 5 for x^2 + 2x + x, I would simply
    #  plug in the value 5 for x and get as x -> 5; y -> 37.

    # For non-continuous functions however, limits can be more difficult. For example, as x -> 0 for sin(x)/x, we cannot
    #  divide by 0 meaning that we will have to take a more brute force approach to find that as x -> 0 y -> 1.

    # It is also common for limits to approach infinity. For example, in 25/x as x -> 0. In this case as x -> 0 (from
    # left); y -> -∞ as x -> 0 (from right); y -> ∞.


def exercises_on_limits():
    print("exercises_on_limits")

    # 1. x -> 0 in (x^2 - 1)/(x-1) = (0 - 1)/(0 - 1) = 1
    #  x -> 0; y -> 1

    # 2. x -> -5 in (x^2 - 25)/(x + 5) = (x-5)(x+5)/(x+5) = x-5
    #  x -> -5; y -> -10

    # 3. x -> 4 in (x^2 - 2x - 8)/(x - 4) = (x + 2)(x - 4)/(x - 4) = x + 2
    #  x -> 4; y -> 6

    # 4. x -> -∞ in 25/x
    #  x -> -∞; y -> 0

    # 5. x -> 0 in 25/x
    #  x -> 0^-; y -> -∞ as x -> 0^+; y -> ∞


def limits_fn():
    intro_to_differential_calculus()
    intro_to_integral_calculus()
    the_method_of_exhaustion()
    # calculus_of_infinitesimals()
    calculus_applications()
    calculating_limits()
    exercises_on_limits()
