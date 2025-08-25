import numpy as np
import numpy.typing as npt

from solver import ODEFunc, enforce_1d


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

    Y_1 = newtons_method_approx1d(g, 1e-5, y_k)

    return y_k + h * f(Y_1)


# Deriving the solution in n-d instead of 1-d

# If we start in 1-d
# Our RK method is:

# Y_1 = y_n + a11 * h * f(Y_1)

# with a11 = 1/2

# relabeling Y_1 to x, we get

# (h/2) * f(x) + y_n - x = 0

# let

# g(x) = (h/2) * f(x) + y_n - x

# then we just run newton's method on g(x) to get our value for Y_1 (aka x)

# ---

# Now let's go to n-d with the same method

# Y_1-1 = y_n-1 + (h/2) * f1(Y_1)
# Y_1-2 = y_n-2 + (h/2) * f2(Y_1)
# ...
# Y_1-d = y_n-d + (h/2) * fd(Y_1)

# or vectorially

# Y_1 = y_n + (h/2) * f(Y_1)

# then if we define (relabelling Y_1 as x

# gi(x) = (h/2) * fi(x) + y_n-i - x-i

# then we want to use newton's method on the system:

# g1(x) = 0
# g2(x) = 0
# ...
# gd(x) = 0

# or vectorially g(x) = 0

# (I wanna make sure I didn't screw up something here since i.e. the functions still take the entirety of x as an argument)

# now let's build up from first principles how to solve this with newton's method

# in 1-d scalar form, we can derive newton's method like so

# f(x) ~= f(x_n) + f'(x_n) * (x - x_n) where x_n is the point we expand about (i.e. initial guess)

# then we just need to solve f(x_n) + f'(x_n) * (x - x_n) = 0, which we can get with rearranging

# x = x_n - f(x_n) / f'(x_n)

# and then we just apply that iteratively

# now let's go to n-d, starting in scalar terms (with x and x_n vectors, x_n is the initial guess and x-i is the ith component of x)

# we use the multivariate chain rule

# f1(x) ~= f1(x_n) + del f1 / del x-1 (x_n) * (x-1 - x_n-1) + del f1 / del x-2 (x_n) * (x-2 - x_n-2) + ... del f1 / del x-d (x_n) * (x-d - x_n-d)
# ...
# fd(x) ~= ...

# notice we can rewrite this as (with . as the dot product)

# f1(x) ~= f1(x_n) + (grad f_1 ) . (x-x_n)
# ...
# fd(x) ~= fd(x_n) + (grad f_d ) . (x-x_n)


# then remember that matrix-vector multiplication M @ v can be written as

# M @ v = [M_1- . v, M_2- . v, ... M_d- . v]^T

# where M_i- is the ith row of M

# so if we define the matrix

# J(x) = [grad f1(x), grad f2(x), ..., grad fd(x)], where grad f1(x) is the first row of our matrix, grad f2(x) is the second row, etc.

# but recall the Jacobian matrix of f(x) is precisely

# [ del f1 / del x1, del f1 del x2...]
# [ del f2 / del x2, ...]
# ...
# [...]

# so J(x) is precisely the Jacobian matrix

# then we can finally write our approximation of f in vectorial form

# f(x) ~= f(x_n) + J(x_n) @ (x - x_n) where J(x_n) is the jacobian matrix evaluated at x_n

# then we want to find

# f(x_n) + J(x_n) @ (x - x_n) = 0

# which we can find with

# x = x_n - J^-1(x_n) @ f(x_n)

# where J^-1 is the inverse of the jacobian matrix

# though practically, it's easier to just solve

# J(x_n) @ (x-x_n) = -f(x_n)

# for (x-x_n) with methods other than inversion, and then extract x by adding x_n


# ---

# now applying this to our problem

# g(x) = 0

# with g(x) = (h/2) * f(x) + y_n - x

# then

# J_g(x_n) = (h/2) * J_f(x_n)  - I_d (where I_d is the dxd Identity matrix)

# then we get

# x = x_n - J_g^-1(x_n) @ g(x_n)

# though practically, it's easier to just solve

# J_g(x_n) @ (x-x_n) = -g(x_n)

# for (x-x_n) with methods other than inversion, and then extract x by adding x_n

# which we do successively until magnitude of g(x_n) is zero to within our tolerance, and thus we have our Y_1 vector finally

# at which point out update is trivially

# y_n+1 = y_n + h * f(Y_1)


# Stage 1 gauss legendre
def implicit_midpoint(y_k, h: np.floating, f: ODEFunc) -> npt.NDArray:
    y_k = enforce_1d(y_k)

    def g(x):
        return (h / 2) * f(x) + y_k - x

    # Y_1 = newtons_method_approx(g, 1e-5, y_k)

    # return y_k + h * f(Y_1)

    pass


def newtons_method_approxnd(the_function, initial_guess, tolerance=1e-5, maxiters=100):

    xn = enforce_1d(initial_guess)
    dim = len(xn)
    thefunc = the_function(xn)
    iters = 0
    while np.linalg.norm(thefunc) >= tolerance and iters < maxiters:
        iters += 1
        Jac = np.zeros((dim, dim))
        s = (10 ** (-8)) * (
            1 + np.abs(xn)
        )  # abs is componentwise so this computes a step size for each element of xn

        # now we compute all the columns of Jac

        # Each column J[:,i] = del f / del xi (xn) ~= f(xn + s * e_i) - f(x) / s
        for i in range(dim):
            e_i = np.zeros(dim)
            e_i[i] = s[i]
            the_deriv = (the_function(xn + e_i) - thefunc) / s[i]
            Jac[:, i] = the_deriv

        # solve J(x_n) @ (x-x_n) = -f(x_n) to get (x-xn), so x = xn + (x-xn)
        # with (x-xn) = solution to J(x_n) @ v = -f(x_n)
        deltax = np.linalg.solve(Jac, -thefunc)
        xn = xn + deltax

        thefunc = the_function(xn)
    return xn


def newtons_method_exact1d(the_function, the_derivative, iterations, initial_guess):
    # f(x) ~= f(x0) + f'(x0) (x-x0) -> zero at f(x0) + f'(x0) (x-x0) = 0 -> x = x0 - f(x0)/f'(x0)
    xn = initial_guess
    for i in range(iterations):
        xn = xn - the_function(xn) / the_derivative(xn)
    return xn


def newtons_method_approx1d(the_function, initial_guess, tolerance=1e-5, maxiters=100):
    assert initial_guess.shape == (1,)
    # f(x) ~= f(x0) + f'(x0) (x-x0) -> zero at f(x0) + f'(x0) (x-x0) = 0 -> x = x0 - f(x0)/f'(x0)
    xn = initial_guess
    thefunc = the_function(xn)
    iters = 0
    while np.abs(thefunc)[0] >= tolerance and iters < maxiters:
        iters += 1
        s = (10 ** (-8)) * (1 + np.abs(xn))
        thederiv = (the_function(xn + s) - thefunc) / s
        xn = xn - thefunc / thederiv
        thefunc = the_function(xn)
    return xn
