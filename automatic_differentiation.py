import matplotlib.pyplot as plt
import numpy as np

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


def automatic_differentiation_fn():
    segment_intro()
    what_automatic_differentiation_is()
    autodiff_with_pytorch()
    autodiff_with_tensorflow()
