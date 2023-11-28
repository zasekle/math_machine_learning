def what_is_linear_algebra():
    print("what_is_linear_algebra() running!")

    # Algebra is arithmetic that include non-numerical entities like x. For example, `2x + 5 = 25`. Linear algebra
    #  is algebra is only used on linear equations. A more formal definition of linear algebra is "Solving for
    #  unknowns within a system of linear equations".

    # xf - xi = v*t
    # Robber, xi = 0
    # x = 150 * t
    # x/150 = t
    # Cop, xi = -(180 * 5/60)
    # x + 15 = 180 * t
    # x/180 + 1/12 = t
    # x/150 = x/180 + 1/12
    # x/150 - x/180 = 1/12
    # (6x - 5x)/900 = 1/12
    # x/900 = 1/12
    # x = 75 km


def plotting_a_system_of_linear_equations():
    import numpy as np
    import matplotlib.pyplot as plt

    t = np.linspace(0, 40, 1000)  # start, finish, n points

    d_r = 2.5 * t
    d_s = 3 * (t - 5)

    fig, ax = plt.subplots()
    plt.title('A Bank Robber Caught')
    plt.xlabel('time (in minutes)')
    plt.ylabel('distance (in km)')
    ax.set_xlim([0, 40])
    ax.set_ylim([0, 100])
    ax.plot(t, d_r, c='green')
    ax.plot(t, d_s, c='brown')
    plt.axvline(x=30, color='purple', linestyle='--')
    _ = plt.axhline(y=75, color='purple', linestyle='--')

    plt.show()


def linear_algebra_exercise():
    import numpy as np
    import matplotlib.pyplot as plt

    t = np.linspace(0, 50, 1000)  # start, finish, n points

    d_a = t
    d_b = 4 * (t - 30)

    fig, ax = plt.subplots()
    plt.title('Solar panels')
    plt.xlabel('time (in days)')
    plt.ylabel('power (in kJ)')
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 70])
    ax.plot(t, d_a, c='green')
    ax.plot(t, d_b, c='brown')
    plt.axvline(x=40, color='purple', linestyle='--')
    _ = plt.axhline(y=40, color='purple', linestyle='--')

    plt.show()


def tensors():
    print("tensors")

    # A scalar is a single value.
    # A vector is a one dimensional array.
    # A matrix is essentially a two dimensional array.
    # A tensor generalizes vectors and matrices to any number of dimensions.


def scalars():
    # scalar properties
    # no dimensions
    # single number
    # denoted in lowercase, italics
    # should be typed, like all other tensors: e.g. int, float32

    x = 25

    print(x)
    print(type(x))

    y = 3.5

    print(type(y))

    # pytorch
    import torch

    x_pt = torch.tensor(25)

    print(x_pt)
    print(x_pt.shape)

    # pytorch and tensorflow seem to be alternatives to each other
    import tensorflow as tf

    x_tf = tf.Variable(25, dtype=tf.int16)

    print(x_tf)
    print(x_tf.shape)

    y_tf = tf.Variable(3, dtype=tf.int16)

    print(x_tf + y_tf)

    tf_sum = tf.add(x_tf, y_tf)

    print(tf_sum)

    print(tf_sum.numpy())

    print(type(tf_sum.numpy()))

    tf_float = tf.Variable(25., dtype=tf.float16)

    print(tf_float)


def data_structures_for_linear_algebra_fn():
    what_is_linear_algebra()
    # plotting_a_system_of_linear_equations()
    # linear_algebra_exercise()
    tensors()
    scalars()
