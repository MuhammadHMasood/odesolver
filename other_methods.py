import numpy as np
import numpy.typing as npt

from solver import ODEFunc


# supposing 1 stage 1-d implicit method we have

# dy/dt = f(y)

# and our method is

# Y_1 = y_n + a11 * h * f(Y_1)

# and then once we have Y1, we can just do

# y_n+1 = y_n + b_1 * h * f(Y_1)

# ---

# to solve Y_1 = y_n + a11 * h * f(Y_1) is equivalent to solving 0  = -Y_1 + y_n + a11 * h * f(Y_1)

# which is equivalent to finding the zero of  -Y_1 + y_n + a11 * h * f(Y_1)

# relabelling Y_1 < - > x we see that we want to find when the function B * f(x) + C - x = 0

# with C = y_n, B = a_11 * h

# so then label g(x) = B * f(x) + C - x

# if we have an explicit analytic form for f'(x), then we get g'(x) = B * f'(x) - 1

# else if we just have f(x), we can do f'(x0) = (f(x0 + s) - f(x0)) / s

# and then use newton's method on g(x)

# with an initial guess of y_n

# then our update is like normal: y_n+1 = y_n + h * f(Y_1)

# for us Y_1 = y_n + (h/2) * f(Y_1)

# so we want to solve g(x) = (h/2) * f(x) + y_n - x = 0


# Stage 1 gauss legendre
def implicit_midpoint_1d(y_k, h: np.floating, f: ODEFunc) -> npt.NDArray:
    assert y_k.shape == (1,)

    def g(x):
        return (h / 2) * f(x) + y_k - x

    Y_1 = newtons_method_approx(g, 1e-5, y_k)

    return y_k + h * f(Y_1)


def newtons_method_exact(the_function, the_derivative, iterations, initial_guess):
    # f(x) ~= f(x0) + f'(x0) (x-x0) -> zero at f(x0) + f'(x0) (x-x0) = 0 -> x = x0 - f(x0)/f'(x0)
    xn = initial_guess
    for i in range(iterations):
        xn = xn - the_function(xn) / the_derivative(xn)
    return xn


def newtons_method_approx(the_function, initial_guess, tolerance=1e-5, maxiters=100):
    # f(x) ~= f(x0) + f'(x0) (x-x0) -> zero at f(x0) + f'(x0) (x-x0) = 0 -> x = x0 - f(x0)/f'(x0)
    xn = initial_guess
    thefunc = the_function(xn)
    iters = 0
    while np.abs(thefunc) >= tolerance and iters < maxiters:
        iters += 1
        s = (10 ** (-8)) * (1 + np.abs(xn))
        thederiv = (the_function(xn + s) - thefunc) / s
        xn = xn - thefunc / thederiv
        thefunc = the_function(xn)
    return xn
