import numpy as np
import numpy.typing as npt

from solver import ODEFunc, enforce_1d


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

    Y_1 = newtons_method_approx1d(g, y_k)

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


def newtons_method_approxnd_1snd_opt(
    h, a11, the_function, initial_guess, tolerance=1e-5, maxiters=100, y_k=None
):

    xn = enforce_1d(initial_guess)
    if y_k is None:
        y_k = xn.copy()
    dim = len(xn)
    thefunc = the_function(xn)
    the_g = h * a11 * thefunc + y_k - xn
    iters = 0
    while np.linalg.norm(the_g) >= tolerance and iters < maxiters:
        iters += 1
        Jac_f = np.zeros((dim, dim))
        s = (10 ** (-8)) * (
            1 + np.abs(xn)
        )  # abs is componentwise so this computes a step size for each element of xn

        # now we compute all the columns of Jac

        # Each column J[:,i] = del f / del xi (xn) ~= f(xn + s * e_i) - f(x) / s
        for i in range(dim):
            e_i = np.zeros(dim)
            e_i[i] = s[i]
            the_deriv = (the_function(xn + e_i) - thefunc) / s[i]
            Jac_f[:, i] = the_deriv

        Jac = h * a11 * Jac_f - np.identity(dim)
        # solve J(x_n) @ (x-x_n) = -f(x_n) to get (x-xn), so x = xn + (x-xn)
        # with (x-xn) = solution to J(x_n) @ v = -f(x_n)
        deltax = np.linalg.solve(Jac, -the_g)
        xn = xn + deltax

        thefunc = the_function(xn)
        the_g = h * a11 * thefunc + y_k - xn
    return xn


# Trivially generalize to any a11 and b1 value (any 1-stage implicit r-k method)
def Implicit_1Stage_RK_method_generator_semianalytical(
    a11, b1, tolerance=1e-5, maxiters=100
):
    def implicit_1stage_nd_rk(y_k, h: np.floating, f: ODEFunc) -> npt.NDArray:
        y_k = enforce_1d(y_k)

        Y_1 = newtons_method_approxnd_1snd_opt(h, a11, f, y_k, tolerance, maxiters)

        return y_k + h * b1 * f(Y_1)

    return implicit_1stage_nd_rk


# Trivially generalize to any a11 and b1 value (any 1-stage implicit r-k method)
def Implicit_1Stage_RK_method_generator(a11, b1, tolerance=1e-5, maxiters=100):
    def implicit_1stage_nd_rk(y_k, h: np.floating, f: ODEFunc) -> npt.NDArray:
        y_k = enforce_1d(y_k)

        def g(x):
            return (h) * a11 * f(x) + y_k - x

        Y_1 = newtons_method_approxnd(g, y_k, tolerance, maxiters)

        return y_k + h * b1 * f(Y_1)

    return implicit_1stage_nd_rk


# Stage 1 gauss legendre nd
def implicit_midpoint(y_k, h: np.floating, f: ODEFunc) -> npt.NDArray:
    y_k = enforce_1d(y_k)

    def g(x):
        return (h / 2) * f(x) + y_k - x

    Y_1 = newtons_method_approxnd(g, y_k)

    return y_k + h * f(Y_1)


# Now onto 2 stage 1 d

# Now we have

# Y_1 = yn + a11 * h * f(Y_1) + a12 * h * f(Y_2)
# Y_2 = yn + a21 * h * f(Y_1) + a22 * h * f(Y_2)

#

# IF WE DEFINE THE K VECTOR AS K = [f(Y_1), f(Y_2)]^T AND THE Y VECTOR AS Y = [Y_1, Y_2]^T,
# yn = [yn, yn]^T and A as the A matrix

# THEN WE CAN WRITE THIS AS

# h * A@f(Y) + yn - Y = 0

# g(Y) =  h * A@f(Y) + yn - Y

# g(Y) = 0

# to find Y

# or in K form we have

#  K = f(Y), and Y = yn + h * A@K

#  so

#  K = f(h * A@K + yn)

#  so

#  g(K) =  f(h * A@K + yn) - K

#  g(K) = 0 to find K


def Implicit_2Stage_RK_method_generator(
    A_matrix, b_vector, tolerance=1e-5, maxiters=100
):
    b_vec = enforce_1d(b_vector)
    assert len(b_vector) == 2
    assert A_matrix.shape == (2, 2)

    def implicit_2stage_1d_rk(y_k, h: np.floating, f: ODEFunc) -> npt.NDArray:
        assert y_k.shape == (1,)

        # this is a 1-d problem so we have to redefine yn (y_k) and f to be 2d
        yn1 = np.full(2, y_k)

        # redefine f to a vector function acting on both stages values
        # This is kind of sloppy because in theory f accepts shape (1,) and outputs shape (1,)
        # but practically 1-d F will generally be define in a way that it works on scalars and (1,) equally
        def f_vec(x):
            return np.array([f(x[0]), f(x[1])])

        # We want to find Y = [Y_1, Y_2]^T
        # Y form
        # Y = yn + h A @ f(Y)
        # g(Y) =  h * A@f(Y) + yn - Y
        # Remember y_k and f are 1-d
        def g(x):
            return h * A_matrix @ f_vec(x) + yn1 - x

        Y = newtons_method_approxnd(g, yn1, tolerance, maxiters)

        return y_k + h * b_vec.T @ f_vec(Y)

    return implicit_2stage_1d_rk


# We make use of the semianalytical evaluation of the Jacobian for g
# recall g(Y) =  h * A@f(Y) + yn - Y
# J_g = h * A @ J_f - I, where I is the identity matrix
# But recall f(Y) = [f(Y1), f(Y2)]
# so the J_f matrix is:
# [D1 f1(Y1), D2 f1(Y1)]
# [D1 f2(Y1), D2 f2(Y1)]
# but since f1 is only a function of Y1,
# and f2 only a func of Y2, then D1 f2 = D2 f1 = 0
# leaving [D1 f1, 0] [0, D2 f2] and D1 f1 = f'(y1), and D2 f2 = f'(y2)
# leaving
# [f'(y1), 0]
# [0, f'(y2)]
def newtons_method_approx_ns1drk_opt(
    A_matrix, h, the_function, initial_guess, tolerance=1e-5, maxiters=100, yn=None
):
    xn = enforce_1d(initial_guess)
    # Usually our initial guess is yn
    if yn is None:
        yn = xn.copy()
    stages = len(xn)  # dim is the number of stages now
    thefunc = np.array([the_function(xn[i]) for i in range(stages)])
    the_g = h * A_matrix @ thefunc + yn - xn
    iters = 0
    while np.linalg.norm(the_g) >= tolerance and iters < maxiters:
        iters += 1
        Jac_f = np.zeros((stages, stages))
        s = (10 ** (-8)) * (
            1 + np.abs(xn)
        )  # abs is componentwise so this computes a step size for each element of xn
        # the initial guesses are all yn (1d) but then they change so we will do it componentwise

        # now we compute all the columns of Jac_f which is simply Jac_f_ik = delta_ik (f'(yi))

        # Each diagonal element J[i,i] = f'(yi) ~= f(yi + s[i]) - f(yi) / s
        for i in range(stages):
            the_deriv = (the_function(xn[i] + s[i]) - thefunc[i]) / s[i]
            Jac_f[i, i] = the_deriv

        # solve J_g(x_n) @ (x-x_n) = -g(x_n) to get (x-xn), so x = xn + (x-xn)
        # with (x-xn) = solution to J_g(x_n) @ v = -g(x_n)
        Jac = h * A_matrix @ Jac_f - np.identity(stages)

        deltax = np.linalg.solve(Jac, -the_g)
        xn = xn + deltax
        thefunc = np.array([the_function(xn[i]) for i in range(stages)])
        the_g = h * A_matrix @ thefunc + yn - xn
    return xn


def Implicit_2Stage_1d_RK_method_generator_semianalytical(
    A_matrix, b_vector, tolerance=1e-5, maxiters=100
):
    b_vec = enforce_1d(b_vector)
    assert len(b_vector) == 2
    assert A_matrix.shape == (2, 2)

    def implicit_2stage_1d_rk(y_k, h: np.floating, f: ODEFunc) -> npt.NDArray:
        assert y_k.shape == (1,)

        # this is a 1-d problem so we have to redefine yn (y_k) and f to be 2d
        yn1 = np.array([y_k.item(), y_k.item()])

        # redefine f to a vector function acting on both stages values
        # This is kind of sloppy because in theory f accepts shape (1,) and outputs shape (1,)
        # but practically 1-d F will generally be define in a way that it works on scalars and (1,) equally
        def f_vec(x):
            return np.array([f(x[0]), f(x[1])])

        # We want to find Y = [Y_1, Y_2]^T
        # Y form
        # Y = yn + h A @ f(Y)
        # g(Y) =  h * A@f(Y) + yn - Y

        Y = newtons_method_approx_ns1drk_opt(A_matrix, h, f, yn1, tolerance, maxiters)

        return y_k + h * b_vec.T @ f_vec(Y)

    return implicit_2stage_1d_rk


def Implicit_nStage_1d_RK_method_generator_semianalytical(
    A_matrix, b_vector, tolerance=1e-5, maxiters=100
):

    b_vec = enforce_1d(b_vector)
    stages = len(b_vector)
    assert A_matrix.shape == (stages, stages)

    def implicit_nstage_1d_rk(y_k, h: np.floating, f: ODEFunc) -> npt.NDArray:
        assert y_k.shape == (1,)

        # this is a 1-d problem so we have to redefine yn (y_k) and f to be 2d
        yn1 = np.tile(y_k.item(), stages)

        # redefine f to a vector function acting on both stages values
        # This is kind of sloppy because in theory f accepts shape (1,) and outputs shape (1,)
        # but practically 1-d F will generally be define in a way that it works on scalars and (1,) equally
        def f_vec(x):
            return np.array([f(x[i]) for i in range(stages)])

        # We want to find Y = [Y_1, Y_2]^T
        # Y form
        # Y = yn + h A @ f(Y)
        # g(Y) =  h * A@f(Y) + yn - Y

        Y = newtons_method_approx_ns1drk_opt(A_matrix, h, f, yn1, tolerance, maxiters)

        return y_k + h * b_vec.T @ f_vec(Y)

    return implicit_nstage_1d_rk


# def Implicit_1Stage_RK_method_generator(a11, b1, tolerance=1e-5, maxiters=100):
#     def implicit_1stage_nd_rk(y_k, h: np.floating, f: ODEFunc) -> npt.NDArray:
#         y_k = enforce_1d(y_k)

#         def g(x):
#             return (h) * a11 * f(x) + y_k - x

#         Y_1 = newtons_method_approxnd(g, y_k, tolerance, maxiters)

#         return y_k + h * b1 * f(Y_1)

#     return implicit_1stage_nd_rk


def Implicit_nStage_nd_RK_method_generator_bruteforce(
    A_matrix, b_vector, tolerance=1e-5, maxiters=100
):
    b_vec = enforce_1d(b_vector)
    stages = len(b_vector)
    assert A_matrix.shape == (stages, stages)

    def implicit_nstage_nd_rk(y_k, h: np.floating, f: ODEFunc) -> npt.NDArray:
        y_k = enforce_1d(y_k)
        dim = len(y_k)

        y_k_stacked = np.tile(y_k, stages)

        def g_stacked(x):
            g_sd = np.zeros(stages * dim)
            for i in range(stages):
                g_i = np.zeros(dim)
                g_i += y_k - x[i * dim : (i + 1) * dim]
                for b in range(stages):
                    Y_b = x[b * dim : (b + 1) * dim]
                    g_i += h * A_matrix[i, b] * f(Y_b)

                g_sd[i * dim : (i + 1) * dim] = g_i
            return g_sd

        Y_stacked = newtons_method_approxnd(g_stacked, y_k_stacked, tolerance, maxiters)

        result = np.zeros(dim)

        for i in range(stages):
            result += b_vec[i] * h * f(Y_stacked[i * dim : (i + 1) * dim])
        result += y_k

        return result

    return implicit_nstage_nd_rk


# we have

# Y_1 = y_n + h * ( (a11) * f(Y_1) + (a12) * f(Y_2) + ... (a1s) * f(Y_s) )
# Y_2 = y_n +  h * ( (a21) * f(Y_1) + (a22) * f(Y_2) + ... (a2s) * f(Y_s) )
# ...
# Y_s = y_n +  h * ( (as1) * f(Y_1) + (as2) * f(Y_2) + ... (ass) * f(Y_s) )

# where each equation is a vector equation (f and Y_i and y_n are vectors)

# if we define the K matrix in the following way:

#  [f(Y_1), f(Y_2), ...]^T

#  but f and Y_i are vectors so that turns into (K is s x d matrix)

# [ f1(Y_1), f2(Y_1), ... fd(Y_1)]
# [ f1(Y_2), f2(Y_2), ... fd(Y_2)]
# [ ...                                        ]
# [ f1(Y_s) f2(Y_s), ... fd(Y_s) ]

# (A is s x s matrix)

# (s x s, s x d -> s x d)

# then A @ K =
# [ a11 f1(Y_1) + a12 f1(Y_2) + ..., a11 f2(Y_1) + a12 f2(Y_2) + ..., ..., a11 fd(Y_1) + a12 fd(Y_2) + ...]
# [...]
# ...
# [...]

# And then the Y matrix is (s x d)

# [Y_1-1, Y_1-2, ..., Y_1-d]
# [Y_2-1, Y_2-2, ..., Y_2-d]
# ...
# [Y_s-1, Y_s-2, ..., Y_s-d]

# then we define the y_n matrix as

# [yn-1, yn-2, ... yn-d]
# ... (s times)
# [yn-1, yn-2, ... yn-d]

# so we can write:

# Y = yn + h * A @ K
# 0 = h * A @ K + yn - Y

# with:

# A: s x s
# K: s x d
# Y: s x d
# f: d
# yn: d expanded into s x d

# in index form
# A_ij = aij
# Y_ij = Y_i-j
# K_ij = fj(Y_i)
# yn_ij = yn-j

# and then our equation is

# Y_ij = yn_ij + h * A_ib * K_bj

# <-> Y_i-j = yn-j + h * aib * fj(Y_b)

# <->  h * aib * fj(Y_b) + yn-j - Y_i-j = 0

# [1]

# now define g_ij = h * aib * fj(Y_b) + yn-j - Y_i-j

# and now let's take the derivative wrt the 2-d y variable, del g_ij / del Y_k-l

# let's go component by component taking advantage of linearity:
# del Y_i-j / del Y_k-l = delta_ik delta _jl
# del yn-j / del Y_k-l = 0
# del  h * aib * fj(Y_b) / del Y_k-l =   h * aib * del fj(Y_b) / del Y_k-l

# now let us use the product rule on del fj(Y_b) / del Y_k-l

# ---
# with x vector, f scalar:

# del f(g(x)) / del xi = del f(g(x))/ del gj(x) * del gj(x) / del xi

# = grad_g f(g(x)) . del g(x) / del xi

# for our case g(x) = [x-1, x-2, x-3, ... x-d] = x

# so we get grad_f(x) . del x / del xi
# ---

# applying that, we get

# del fj(Y_b) / del Y_k-l = grad fj(Y_b) . del Y_b / del Y_k-l

# del Y_b / del Y_k-l = delta_bk * e_l
# (because del Y_b / del_Y_k-l is 0 if b !=k which gives the delta_bk, and if they are the same, then you get [0,...,0,1,0,...,0] where the 1 is at the l-th index, i.e. e_l

# so del fj(Y_b) / del Y_k-l = grad fj(Y_b) . (delta_bk * e_l) = (delta_bk * grad fj(Y_b)) . e_l

# but grad fj(Y_b) = [del fj / del x1, del fj / del x2, ...], so e_l . grad fj(Y_b) = del fj / del xl (Y_b) (where xl is just the l-th argument of fi)

# so we get

# del fj(Y_b) / del Y_k-l = delta_bk * del fj / del xl (Y_b)

# so thus we get

# del g_ij / del Y_k-l =  h * aib * delta_bk * del fj / del xl (Y_b) - delta_ik delta _jl = h * aik * del fj / del xl (Y_b) - delta_ik delta _jl

# and for checking we can see 4 free indices on each term (i,k,j,l)

# for a fixed ik, we see that this gives us a Jacobian matrix of f in j and l evaluated at Y_b, possibly minus the identity matrix if i = k

# notice that ik are the 'stage indices' and 'jl' are the 'dimensional indices'

# Intuitively we also can see that this g'_ikjl object is something that if you 'matrix dot' it with a matrix (2 index object, where we contract over both indices), gives out a matrix (2 remaining indices)

# Now we just need to use our calculation of g' to write g as a linear approximation g(x) ~= g(x0) + g'(x0) (x-x0)

# g_ij = h * aib * fj(Y_b) + yn-j - Y_i-j
# g_i = h * (sum_b=1^s a_ib f(Y_b)) + yn - Y_i

# (where a_ib = aib but is clearer looking)

# now there's a derivation you can do with the frechet derivative but you can kind of intuitively see that we can write it like this:

# g_stacked(Y) ~= g_stacked(Y_b) + Dg(Y_b) delta Y_stacked

# where g_stacked is [g_1, g_2, g_3,...g_s]
# and Y_stacked is [Y_1-1, Y_1-2,...Y_1-d, Y_2-1, Y_2-2, ... Y_s-1, ... Y_s-d]

# [1]: we can check this is correct by looking at this:

# A @ K =
# [ a11 f1(Y_1) + a12 f1(Y_2) + ..., a11 f2(Y_1) + a12 f2(Y_2) + ..., ..., a11 fd(Y_1) + a12 fd(Y_2) + ...]
# [...]
# ...
# [...]

# and seeing how (A @ K)_ij is indeed A_ib * K_bj


def Implicit_nStage_nd_RK_method_generator_semi_analytical(A_matrix, b_vector):
    # TODO
    pass
