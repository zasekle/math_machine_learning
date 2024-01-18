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


def derivatives_and_differentiation_fn():
    the_delta_method()
