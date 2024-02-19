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


def advanced_partial_derivative_exercises():
    print("advanced_partial_derivative_exercises")

    # Find all partial derivatives of the following functions.
    # 1) z = y^3 + 5xy
    # 2) The surface area of a cylinder is described by a = 2*pi*r^2 + 2*pi*r*h.
    # 3) The volume of a square prism with a cube cut out of its center is described by v = x^2*y - z^3.

    # z = y^3 + 5xy
    #  dz/dx = 5y
    #  dz/dy = 3y^2 + 5x
    # a = 2*pi*r^2 + 2*pi*r*h
    #  da/dr = 4*pi*r + 2*pi*h
    #  da/dh = 2*pi*r
    # v = x^2*y - z^3
    #  dv/dx = 2*x*y
    #  dv/dy = x^2
    #  dv/dz = -3z^2


def partial_derivative_notation():
    print("partial_derivative_notation")

    # Several common ways of listing partial derivatives are listed below.
    # z = f(x, y)
    # dz/dx
    # df/dx
    # fx
    # Dxf


def the_chain_rule_for_partial_derivatives():
    print("the_chain_rule_for_partial_derivatives")

    # If there is a nested function where y = f(u) and u = g(x) then the chain rule for full derivatives would be:
    #  dy/dx = dy/du * du/dx
    # With univariate functions, the partial derivative is identical:
    #  dy/dx = dy/du * du/dx
    # For example, lets say that there are two different functions y = f(u) and u = g(x, z) this means that:
    #  dy/dx = dy/du * du/dx
    #  dy/dz = dy/du * du/dz


def exercises_on_the_multivariate_chain_rule():
    print("exercises_on_the_multivariate_chain_rule")

    # 1) y=f(u,v), u=g(x), v=h(z)
    #  dy/dx dy/dz
    # 2) y=f(u,v), u=g(x), v=h(x,z)
    #  dy/dx dy/dz
    # 3) y=f(u,v,w), u=g(x), v=h(x), w=j(x)
    #  dy/dx


def point_by_point_regression():
    print("point_by_point_regression")

    xs = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7.])
    ys = torch.tensor([1.86, 1.31, .62, .33, .09, -.67, -1.23, -1.37])

    def regression(my_x, my_m, my_b):
        return my_m * my_x + my_b

    m = torch.tensor([0.9]).requires_grad_()
    b = torch.tensor([0.1]).requires_grad_()

    i = 7
    x = xs[i]
    y = ys[i]

    # This is a pretty arbitrary estimate.
    yhat = regression(x, m, b)

    def squared_error(my_yhat, my_y):
        return (my_yhat - my_y) ** 2

    C = squared_error(yhat, y)

    C.backward()

    # The partial derivative of C with respect to m(dC/dm)
    print(m.grad)

    # The partial derivative of C with respect to b(dC/db)
    print(b.grad)


def the_gradient_of_quadratic_cost():
    print("the_gradient_of_quadratic_cost")

    # Starting with the equation y = mx + b.
    # When starting with the cost of the function (the error) the function used is C = (yhat - y)^2. The actual
    #  functions that we are interested in however, are dC/db and dC/dm. So a simple derivation is as follows.
    #  C = (yhat - y)^2
    #   dC/dyhat = 2(yhat - y)
    #  yhat = mx + b
    #   dyhat/db = 1
    #   dyhat/dm = x
    #  dC/dyhat * dyhat/db = dC/db = 2(yhat - y)
    #  dC/dyhat * dyhat/dm = dC/dm = 2x(yhat - y)
    #  These derivations are what are happening internally in the previous section. It is how the libraries actually use
    #   auto differentiation to calculate stuff.

    # ∇C is the gradient of cost, it is a vector of all the partial derivatives of C with respect to each of the
    #  individual model parameters.  In the above case there are only two parameters, so
    #  ∇C = [dC/db, dC/dm]^(transpose).


def descending_the_gradient_of_cost():
    print("descending_the_gradient_of_cost")

    xs = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    ys = torch.tensor([1.86, 1.31, .62, .33, .09, -.67, -1.23, -1.37])

    def regression(my_x, my_m, my_b):
        return my_m * my_x + my_b

    m = torch.tensor([0.9]).requires_grad_()
    b = torch.tensor([0.1]).requires_grad_()

    # This is a pretty arbitrary estimate.
    yhat = regression(xs, m, b)

    def mean_squared_error(my_yhat, my_y):
        sigma = torch.sum((my_yhat - my_y)**2)
        return sigma/len(my_y)

    cost = mean_squared_error(yhat, ys)

    # Use autodiff to calculate gradient of cost with respect to parameters.
    cost.backward()

    print(m.grad)
    print(b.grad)


def partial_derivative_calculus_fn():
    # what_partial_derivatives_are()
    partial_derivative_exercises()
    calculating_partial_derivatives_with_autodiff()
    advanced_partial_derivatives()
    partial_derivative_notation()
    the_chain_rule_for_partial_derivatives()
    exercises_on_the_multivariate_chain_rule()
    point_by_point_regression()
    the_gradient_of_quadratic_cost()
    descending_the_gradient_of_cost()
