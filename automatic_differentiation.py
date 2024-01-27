import matplotlib.pyplot as plt

import torch
import tensorflow as tf


def segment_intro():
    print("segment_intro")


def what_automatic_differentiation_is():
    print("what_automatic_differentiation_is")

    # The two ways to differentiate that have been covered are listed below.
    #   Numerical diff: The delta method where smaller and smaller deltas are chosen. This introduces rounding error.
    #   Symbolic diff: These are the algebraic rules that were covered in the last section. Computationally inefficient.

    # Automatic differentiation is used for machine learning systems. This handles functions with many inputs and
    #  higher-order derivatives. Automatic differentiation works by applying the chain rule to a sequence of arithmetic
    #  operations. This means that the autodiff proceeds from the outermost function inward when using the chain rule
    #  compared to doing it by hand which goes from the innermost function outward.

    # Aliases for automatic differentiation are.
    #   autodiff
    #   autograd
    #   computational diff.
    #   reverse mode diff.
    #   algorithmic diff.


def autodiff_with_pytorch():
    # TensorFlow and PyTorch are the two most popular automatic differentiation libraries.

    # calculating dy/dx at x = 5 where y = x^2 (the answer is 10)

    x = torch.tensor(5.0)

    x.requires_grad_()  # track forward pass

    y = x ** 2

    # The way that this works is that the requires_grad_() flag is set to true above. This saves all operations done to
    #  the tensor. After that, an operation (x ** 2) is done on the tensor. Finally, the backward() method is called
    #  which will apply the chain rule to all operations (staring at the last operation and moving backwards) that have
    #  been done to this tensor. This will allow the gradient to be calculated.
    y.backward()  # use autodiff

    print(x.grad)


def autodiff_with_tensorflow():
    # calculating dy/dx at x = 5 where y = x^2 (the answer is 10)

    x = tf.Variable(5.0)

    with tf.GradientTape() as t:
        t.watch(x)  # track forward pass
        y = x ** 2

    print(t.gradient(y, x))  # use autodiff


def regression(my_x, my_m, my_b):
    return my_m * my_x + my_b


def regression_plot(my_x, my_y, my_m, my_b):
    fig, ax = plt.subplots()

    ax.scatter(my_x, my_y)

    x_min, x_max = ax.get_xlim()
    y_min = regression(x_min, my_m, my_b).detach().item()
    y_max = regression(x_max, my_m, my_b).detach().item()

    ax.set_xlim([x_min, x_max])
    _ = ax.plot([x_min, x_max], [y_min, y_max])

    plt.show()


def fitting_a_line_with_machine_learning():
    # Using pytorch automatic differentiation library to fit a straight line to data points. y = mx + b

    x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7.])

    # this will simulate a y value with some random noisy to be more realistic
    y = -0.5 * x + 2 + torch.normal(mean=torch.zeros(8), std=0.2)

    # set up data points that the model will need to do some work for
    m = torch.tensor([0.9]).requires_grad_()
    b = torch.tensor([0.1]).requires_grad_()

    regression_plot(x, y, m, b)


def machine_learning_with_autodiff():

    # Note that in a real situation you want a huge number of points here because that will give the algorithm higher
    #  precision in the final value.
    x = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7.])

    # this will simulate a y value with some random noisy to be more realistic
    y = -0.5 * x + 2 + torch.normal(mean=torch.zeros(8), std=0.2)

    # set up data points that the model will need to do some work for
    m = torch.tensor([0.9]).requires_grad_()
    b = torch.tensor([0.1]).requires_grad_()

    optimizer = torch.optim.SGD([m, b], lr=0.01)

    epochs = 1000
    for i in range(epochs):
        # reset gradients to zero; else they accumulate
        optimizer.zero_grad()

        # Step 1: Forward pass to calculate what y is with the current values for m and b.
        yhat = regression(x, m, b)

        def mean_squared_error_loss(my_yhat, my_y):
            sigma = torch.sum((my_yhat - my_y) ** 2)
            return sigma / len(my_y)

        # Step 2: Calculate cost meaning how close the regression is to the true points.
        cost = mean_squared_error_loss(yhat, y)

        # Step 3: Use automatic differentiation to calculate the gradient of cost with respect to initial parameters.
        #  This will show which direction the values for m and b need to go in order to get closer. NOTE: If this gives
        #  direction of the change, what gives magnitude of the change?
        cost.backward()

        # Step 4: Gradient descent, so move along the gradient towards the minimum. NOTE: The lr parameter seems to have
        #  something to do with the magnitude I mentioned above. However, we aren't going into gradient descent yet.
        optimizer = torch.optim.SGD([m, b], lr=0.01)

        optimizer.step()

    regression_plot(x, y, m, b)


def automatic_differentiation_fn():
    segment_intro()
    what_automatic_differentiation_is()
    autodiff_with_pytorch()
    autodiff_with_tensorflow()
    # fitting_a_line_with_machine_learning()
    machine_learning_with_autodiff()
