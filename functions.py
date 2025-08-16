from solver import np, npt

from numpy import sin, cos


def f_exponential_system(y) -> npt.NDArray:
    """
    dy/dt = y, with solution y0 * exp(t)
    The first element is time, dt/dt = 1
    """
    return np.concatenate(([1], y[1:]))


def f_exponential(y) -> npt.NDArray:
    """dy/dt = y"""
    return y


def f_logistic(y) -> npt.NDArray:
    """dy/dt = y * (1 - y)"""
    return y * (1 - y)


a = 2  # pendulum length [m]
m = 1  # mass [kg]
g = 10  # gravitational acceleration [m/s^2]


def f_double_pendulum_v0(y) -> npt.NDArray:
    """It's the double pendulum

    Args:
        y (_type_): [t, theta, phi, theta dot, phi dot]

    Returns:
        npt.NDArray: f(y) = [1, theta dot, phi dot, theta double dot, phi double dot]
    """
    # Theta dot = A, Phi dot = B, Theta = C, Phi = D
    t = y[0]
    C = y[1]
    D = y[2]
    A = y[3]
    B = y[4]

    result = np.zeros(5)
    # A dot = theta double dot
    result[3] = (
        (-3 / 8)
        * (
            (1 / ((16 - 9 * (cos(C - D) ** 2)) / 48))
            * (
                (3 / 16) * (B**2) * sin(C - D) * cos(C - D)
                + (1 / 2) * (A**2) * sin(C - D)
                + (9 / (16 * a)) * g * sin(C) * cos(C - D)
                - (1 / (2 * a)) * g * sin(D)
            )
        )
        * cos(C - D)
        - (3 / 8) * (B**2) * sin(C - D)
        - ((9 * g) / (8 * a)) * sin(C)
    )
    # B dot = phi double dot
    result[4] = (1 / ((16 - 9 * (cos(C - D) ** 2)) / 48)) * (
        (3 / 16) * (B**2) * sin(C - D) * cos(C - D)
        + (1 / 2) * (A**2) * sin(C - D)
        + (9 / (16 * a)) * g * sin(C) * cos(C - D)
        - (1 / (2 * a)) * g * sin(D)
    )
    # C dot = theta dot = A
    result[1] = A
    # D dot = phi dot = B
    result[2] = B

    result[0] = 1

    return result


a = 2.0
g = 10.0


def f_double_pendulum_v1(y) -> npt.NDArray:
    """State y = [t, theta=C, phi=D, theta_dot=A, phi_dot=B]
    Returns dy/dt = [1, Cdot, Ddot, Adot, Bdot]
    """
    t, C, D, A, B = y

    delta = C - D
    s = sin(delta)
    c = cos(delta)
    inv = 48.0 / (16.0 - 9.0 * (c**2))  # = 1 / ((16 - 9 cos^2)/48)

    # B dot (phi double dot)
    Bdot = inv * (
        (3.0 / 16.0) * (B**2) * s * c
        + 0.5 * (A**2) * s
        + (9.0 / (16.0 * a)) * g * sin(C) * c
        - (1.0 / (2.0 * a)) * g * sin(D)
    )

    # A dot (theta double dot): -(3/8)*(Bdot*c + B^2*s + 3(g/a) sin C)
    Adot = -(3.0 / 8.0) * (Bdot * c + (B**2) * s + 3.0 * (g / a) * sin(C))

    result = np.empty(5, dtype=float)
    result[0] = 1.0  # t dot
    result[1] = A  # theta dot
    result[2] = B  # phi dot
    result[3] = Adot  # theta double dot
    result[4] = Bdot  # phi double dot
    return result


def logistic_solution(t, y0):
    """Analytical solution for dy/dt = y * (1 - y)"""
    A = (1 - y0) / y0
    return 1 / (A * np.exp(-t) + 1)


def exponential_solution(t, y0):
    """Analytical solution for dy/dt = y"""
    return y0 * np.exp(t)
