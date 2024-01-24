import matplotlib.pyplot as plt
import numpy as np


def the_delta_method():
    print("the_delta_method")

    # The definition of the derivative.
    # dx/dy = lim Δx->0 (f(x + Δx) - f(x))/Δx

    # If I have a line of y = x^2 + 2x + 2, I can pick two points on this line say (2, 10) and (5, 37) and I can find
    #  the slope between them of 9 = (37-10)/(5-2). Then another point closer to 2 can be picked and another slope can
    #  be found. The closer the point, the closer to the 'actual' slope of the tangent line at x=2. Differentiation will
    #  give the way to find this exact slope of the tangent line.

    def f(my_x):
        my_y = my_x ** 2 + 2 * my_x + 2
        return my_y

    # start, finish, n points
    x = np.linspace(-10, 10, 1000)

    y = f(x)

    m = (37 - 10) / (5 - 2)
    b = 37 - m * 5

    line_y = m * x + b

    x_close = 2.1
    y_close = 10.61
    m_close = (y_close - 10) / (x_close - 2)
    b_close = y_close - m_close * x_close

    line_y_close = m_close * x + b_close

    fig, ax = plt.subplots()
    plt.axvline(x=0, color='lightgray')
    plt.axhline(y=0, color='lightgray')
    plt.ylim(-5, 150)
    plt.scatter(2, 10)
    plt.scatter(5, 37, c='orange', zorder=3)
    plt.plot(x, line_y, c='orange')
    plt.scatter(x_close, y_close, c='blue', zorder=3)
    plt.plot(x, line_y_close, c='blue')
    _ = ax.plot(x, y)

    plt.show()


def how_derivatives_arise_from_limits():
    print("how_derivatives_arise_from_limits")

    # He goes through a simple derivation and example of
    # dx / dy = lim Δx->0 (f(x + Δx) - f(x)) / Δx


def derivative_notation():
    print("derivative_notation")

    # popular method for a derivative include
    # y' f'(x) dy/dx df(x)/dx Dxf
    # y'' f''(x) d^2y/d^2x d^2f(x)/d^2x D^2xf


def the_derivative_of_a_constant():
    print("the_derivative_of_a_constant")

    # assuming c is a constant, then dc/dx = 0


def the_power_rule():
    print("the_power_rule")

    # this is the most basic rule, it says that dx^n/dx = nx^(n-1)
    # e.g. x^3 -> 3x^2


def the_constant_multiple_rule():
    print("the_constant_multiple_rule")

    # d(c*y)/dx = c * dy/dx


def the_sum_rule():
    print("the_sum_rule")

    # d(y + w)/dx = dy/dx + dw/dx


def exercises_on_derivative_rules():
    print("exercises_on_derivative_rules")

    # -15x^2
    # 4x + 2
    # 50x^4 - 18x - 1
    # 2 * 2 + 2 = 6
    # 2 * -1 + 2 = 0


def the_product_rule():
    print("the_product_rule")

    # d(wz)/dx = wdz/dx + zdw/dx


def the_quotient_rule():
    print("the_quotient_rule")

    # d(w/z)/dx = (zdw/dx - wdz/dx)/z^2


def the_chain_rule():
    print("the_chain_rule")

    # dy/dx = dy/du * du/dx


def advanced_exercises_on_derivative_rules():
    print("advanced_exercises_on_derivative_rules")

    # 1. (2x^2 + 6x)*(6x^2 + 10x) + (2x^3 + 5x^2)*(4x + 6)
    # 12x^4 + 20x^3 + 36x^3 + 60x^2 + 8x^4 + 12x^3 + 20x^3 + 30x^2
    # 20x^4 + 88x^3 + 90x^2
    # 2. ((2 - x) * 12x - 6x^2 * -1)/(2 - x)^2
    # (24x - 12x^2 + 6x^2)/(2 - x)^2
    # (-6x^2 + 24x)/(2 - x)^2
    # 3. (2 * (3x + 1)) * 3
    # 3 * (6x + 2)
    # 18x + 6
    # 4. (6 * (x^2 + 5x)^5) * (2x + 5)
    # 5. dy/du * du/dx
    # dy/du * (du/dw * dw/dx)
    # w = x^4 + 1
    # u = (x^4 + 1)^5 + 7 = w^5 + 7
    # -1 * ((x^4 + 1)^5 + 7)^-2 * (5 * (x^4 + 1)^4 * 4x^3)


def the_power_rule_on_a_function_chain():
    print("the_power_rule_on_a_function_chain")

    # du^n/dx = nu^(n-1) * du/dx


def derivatives_and_differentiation_fn():
    the_delta_method()
    how_derivatives_arise_from_limits()
    derivative_notation()
    the_derivative_of_a_constant()
    the_power_rule()
    the_constant_multiple_rule()
    the_sum_rule()
    exercises_on_derivative_rules()
    the_product_rule()
    the_quotient_rule()
    the_chain_rule()
    advanced_exercises_on_derivative_rules()
    the_power_rule_on_a_function_chain()
