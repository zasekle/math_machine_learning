import matplotlib.pyplot as plt

import numpy as np
import torch
import tensorflow as tf
import math  # for constant pi


def f(my_x, my_y):
    return my_x ** 2 - my_y ** 2


def what_partial_derivatives_are():
    # Even in a simple multivariate function such as y = mx + b, y is a function of multiple variables. Therefore, we
    #  can't calculate the full derivative dy/dm or dy/db.
    # Consider the equation z = x^2 - y^2. The partial derivative of dz/dx is obtained by considering y to be a
    #  constant. So dz/dx = 2x - 0 = 2x.

    xs = np.linspace(-3, 3, 1000)

    # must hold y constant at y = 0
    z_wrt_x = f(xs, 0)

    # calculate the value of dz/dx
    def delz_delx(my_x, my_y):
        return 2 * my_x

    x_samples = [-2, -1, 0, 1, 2]

    colors = ['red', 'orange', 'green', 'blue', 'purple']

    def point_and_tangent_wrt_x(my_xs, my_x, my_y, my_f, fprime, col):
        my_z = my_f(my_x, my_y)  # z = f(x, y)
        plt.scatter(my_x, my_z, c=col, zorder=3)

        tangent_m = fprime(my_x, my_y)  # Slope is partial derivative of f(x, y) w.r.t. x
        tangent_b = my_z - tangent_m * my_x  # Line is z=mx+b, so b=z-mx
        tangent_line = tangent_m * my_xs + tangent_b

        plt.plot(my_xs, tangent_line, c=col, linestyle='dashed', linewidth=0.7, zorder=3)

    fig, ax = plt.subplots()
    plt.axvline(x=0, color='lightgray')
    plt.axhline(y=0, color='lightgray')

    for i, x in enumerate(x_samples):
        point_and_tangent_wrt_x(xs, x, 0, f, delz_delx, colors[i])

    plt.ylim(-1, 9)
    plt.xlabel('x')
    plt.ylabel('z', rotation=0)
    _ = ax.plot(xs, z_wrt_x)

    plt.show()

    # Likewise the partial derivative of the above equation is dz/dy = -2y which ends up as an inverted parabola.


def partial_derivative_exercises():
    print("partial_derivative_exercises")

    # z = x^2 - y^2
    # dz/dx = 2x
    # dz/dy = -2y
    # 1) x = 3, y = 0
    #  z = 9; dz/dx = 6; dz/dy = 0
    # 2) x = 2, y = 3
    #  z = -5; dz/dx = 4; dz/dy = -6
    # 3) x = -2, y = -3
    #  z = -5; dz/dx = -4; dz/dy = 6


def calculating_partial_derivatives_with_autodiff():
    x = torch.tensor(3.).requires_grad_()
    y = torch.tensor(0.).requires_grad_()

    # gradient tracking is contagious (it flows forward to z)
    z = f(x, y)

    z.backward()

    print(x.grad)
    print(y.grad)


def advanced_partial_derivatives():
    print("advanced_partial_derivatives")

    # We can get a physical representation of what partial derivatives mean as well. For example, with the formula for
    #  the area of a cylinder v = pi*r^2*l then dv/dl = pi*r^2. This means that when r is held constant (say r = 3) we
    #  can find how much a change in l corresponds to a change in v.

    def cylinder_vol(my_r, my_l):
        return math.pi * my_r ** 2 * my_l

    r = torch.tensor(3.).requires_grad_()
    l = torch.tensor(5.).requires_grad_()

    v = cylinder_vol(r, l)

    print(v)

    v.backward()

    # This means that for every one unit l changes, v changes by 28.27 units^3.
    print(l.grad)

    # For example, with an extra unit for l, v will come out to 141 + 28 = 169. Remember that in this example it is only
    #  true because l is not part of the equation. So with dv/dr = 2*pi*r*l, r is part of the equation and so the 2nd
    #  derivative would need to be used for a constant value.
    print(cylinder_vol(3, 6))


def partial_derivative_calculus_fn():
    # what_partial_derivatives_are()
    partial_derivative_exercises()
    calculating_partial_derivatives_with_autodiff()
    advanced_partial_derivatives()
