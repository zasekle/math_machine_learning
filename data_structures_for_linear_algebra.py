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


def data_structures_for_linear_algebra_fn():
    what_is_linear_algebra()
    # plotting_a_system_of_linear_equations()
