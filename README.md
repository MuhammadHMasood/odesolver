# ODE Solver in Python

This project is a Python-based implementation of various numerical methods for solving ordinary differential equations (ODEs). It provides a flexible framework for defining ODEs and solving them with different one-step methods, including explicit and implicit Runge-Kutta schemes.

## Project Structure

- `main.py`: The main entry point for running examples.
- `solver.py`: Contains the core solver logic (`ODESolve`) and protocol definitions for ODE functions and numerical methods.
- `functions.py`: Defines various ODE functions to be solved, such as `f_exponential` and the equations of motion for a double pendulum.
- `grapher.py`: Includes utilities for plotting and animating the solutions.
- `explicit_rkgen.py`: A generator for explicit Runge-Kutta methods.
- `other_methods.py`: Implementations and derivations for implicit methods, which rely on root-finding algorithms like Newton's method.

## Core Concepts

The solver is built around a few key components defined in `solver.py`:

- **`ODEFunc`**: A protocol for the function `f` in the ODE `dy/dt = f(y)`. It's a callable that takes a NumPy array `y` and returns a NumPy array representing the derivative.
- **`OneStepNumericalMethod`**: A protocol for any one-step solver (like Euler or RK4). It's a callable that takes the current state `y_k`, step size `h`, and the `ODEFunc` `f`, and returns the next state `y_{k+1}`.
- **`ODESolve`**: The main loop that iteratively calls a `OneStepNumericalMethod` to compute the solution trajectory over a given time interval.

---

## Derivations and Implementation Details

This section synthesizes the design and mathematical derivations detailed in the source code comments.

### Explicit Runge-Kutta Methods

Explicit Runge-Kutta (RK) methods are a family of one-step solvers where the solution at the next time step is an explicit function of the current state. They are defined by a Butcher tableau, consisting of a matrix `A` and a vector `b`.

A generic `s`-stage explicit RK method is given by:

1. **Stage Calculations:**
    $$
    \begin{align*}
    K_1 &= f(y_n) \\
    K_2 &= f(y_n + h \cdot a_{21}K_1) \\
    K_3 &= f(y_n + h \cdot (a_{31}K_1 + a_{32}K_2)) \\
    &\vdots \\
    K_s &= f(y_n + h \cdot \sum_{j=1}^{s-1} a_{sj}K_j)
    \end{align*}
    $$

2. **Final Solution:**
    $$
    y_{n+1} = y_n + h \cdot \sum_{i=1}^{s} b_i K_i
    $$

For an explicit method, the matrix `A` is strictly lower triangular, meaning the calculation of each stage `K_i` only depends on the preceding stages `K_j` where `j < i`.

#### Implementation: `ExplicitRK_method_generator`

The `explicit_rkgen.py` file provides a factory function, `ExplicitRK_method_generator`, that takes an `A` matrix and a `b` vector and returns a corresponding `OneStepNumericalMethod`.

- The generator first validates that the `A` matrix is strictly lower triangular.
- The returned `RK_method` function implements the logic for solving a `d`-dimensional ODE system.
- Inside the method, it iterates from `i = 1` to `s` (the number of stages):
  - It calculates the argument for `f` for the current stage `i`:
    $$ Y_i = y_n + h \cdot \sum_{j=1}^{i-1} A_{ij} K_j $$
    This is implemented as `y_k + h * A[i, :i] @ K[:i, :]`. The slicing `A[i, :i]` and `K[:i, :]` elegantly handles the sum over previous stages.
  - It then computes the stage derivative:
    $$ K_i = f(Y_i) $$
    This is stored in the `i`-th row of the `K` matrix.
- Finally, it computes the next step `y_{n+1}` using the `b` vector:
    $$ y_{n+1} = y_n + h \cdot (b^T K) $$
    This is implemented as `y_k + h * b @ K`.

### Implicit Runge-Kutta Methods

Implicit methods are more complex because the calculation of each stage `K_i` can depend on itself and other stages, leading to a system of (usually nonlinear) equations that must be solved.

#### 1-Stage Implicit RK Method (e.g., Implicit Midpoint)

A 1-stage implicit method is defined by:
$$
\begin{align*}
Y_1 &= y_n + h \cdot a_{11} f(Y_1) \\
y_{n+1} &= y_n + h \cdot b_1 f(Y_1)
\end{align*}
$$
The first equation is the challenge. To find `Y_1`, we need to solve a root-finding problem. Let's define a function `g(x)` (where `x` is our candidate for `Y_1`):
$$ g(x) = y_n + h \cdot a_{11} f(x) - x = 0 $$
We can solve `g(x) = 0` using Newton's method.

#### Newton's Method for Systems

Newton's method for a scalar function `g(x)` is an iterative process:
$$ x_{k+1} = x_k - \frac{g(x_k)}{g'(x_k)} $$
For a vector-valued function `g(x)` where `x` is a vector, the update rule becomes:
$$ x_{k+1} = x_k - [J_g(x_k)]^{-1} g(x_k) $$
where `J_g` is the Jacobian matrix of `g`. In practice, it's more numerically stable to solve the linear system:
$$ J_g(x_k) \cdot \Delta x = -g(x_k) $$
and then update `x_{k+1} = x_k + \Delta x`.

For our 1-stage implicit method, `g(x) = y_n + h \cdot a_{11} f(x) - x`. The Jacobian `J_g` is:
$$ J_g(x) = h \cdot a_{11} J_f(x) - I $$
where `J_f` is the Jacobian of the ODE function `f`, and `I` is the identity matrix.

The file `other_methods.py` implements this using `newtons_method_approxnd`, which approximates the Jacobian `J_g` numerically. This is a "brute force" but general approach.

#### N-Stage Implicit RK Methods

For a general `s`-stage implicit method, we have a system of `s` vector equations:
$$
Y_i = y_n + h \sum_{j=1}^{s} a_{ij} f(Y_j) \quad \text{for } i = 1, \dots, s
$$
where each `Y_i` and `y_n` are `d`-dimensional vectors. This creates a coupled system of `s * d` nonlinear equations that must be solved simultaneously to find the stage values `Y_1, ..., Y_s`.

This can be formulated as a single large root-finding problem. Let's define a function for the `i`-th stage equation:
$$
g_i(Y_1, \dots, Y_s) = y_n + h \sum_{j=1}^{s} a_{ij} f(Y_j) - Y_i = 0
$$
We need to solve the system `g_1 = 0, g_2 = 0, \dots, g_s = 0`. To handle this with a standard numerical solver, we can "stack" the `s` stage vectors (each of dimension `d`) into one large vector of size `s * d`:
$$
\mathbf{Y} = [Y_1^T, Y_2^T, \dots, Y_s^T]^T
$$
Define the stacked stage vector and its companion pieces:
$$
\mathbf{Y} = [Y_1^T, Y_2^T, \dots, Y_s^T]^T,\qquad \mathbf{y}_n = [y_n^T, y_n^T, \dots, y_n^T]^T,
$$
$$
F(\mathbf{Y}) = [f(Y_1)^T, f(Y_2)^T, \dots, f(Y_s)^T]^T.
$$
Replace each scalar entry \(a_{ij}\) of the Butcher matrix by the block \(a_{ij} I_d\) to form the block matrix
$$
\mathcal{A} = \begin{bmatrix}
a_{11} I_d & a_{12} I_d & \cdots & a_{1s} I_d \\
a_{21} I_d & a_{22} I_d & \cdots & a_{2s} I_d \\
\vdots & \vdots & \ddots & \vdots \\
a_{s1} I_d & a_{s2} I_d & \cdots & a_{ss} I_d
\end{bmatrix}.
$$
Then the coupled system is the single stacked equation
$$
G(\mathbf{Y}) = \mathbf{y}_n + h\, \mathcal{A} F(\mathbf{Y}) - \mathbf{Y} = 0.
$$

There are two main approaches implemented to solve this system:

1. **Brute-Force with Numerical Jacobian:** `Implicit_nStage_nd_RK_method_generator_bruteforce` builds `g_stacked(\mathbf{Y}) = G(\mathbf{Y})` and calls the generic `newtons_method_approxnd`, which finite-differences every column of the full \((sd)\times(sd)\) Jacobian. Simple; lots of function evaluations.


2. **Semi-Analytical Jacobian:** Differentiate the i-th block equation
$$
g_i = y_n + h \sum_{j=1}^s a_{ij} f(Y_j) - Y_i
$$
with respect to \(Y_k\):
$$
\frac{\partial g_i}{\partial Y_k} = h a_{ik} J_f(Y_k) - \delta_{ik} I_d.
$$
Hence the full Jacobian has block structure
$$
J_G(\mathbf{Y}) = \begin{bmatrix}
h a_{11} J_f(Y_1) - I_d & h a_{12} J_f(Y_2) & \cdots & h a_{1s} J_f(Y_s) \\
h a_{21} J_f(Y_1) & h a_{22} J_f(Y_2) - I_d & \cdots & h a_{2s} J_f(Y_s) \\
\vdots & \vdots & \ddots & \vdots \\
h a_{s1} J_f(Y_1) & h a_{s2} J_f(Y_2) & \cdots & h a_{ss} J_f(Y_s) - I_d
\end{bmatrix}.
$$
`newtons_method_approx_ns_nd_rk_opt` computes each small \(J_f(Y_k)\) (size \(d\times d\)) once per iteration via finite differences, assembles this matrix, solves
$$
J_G(\mathbf{Y}^{(m)}) \Delta \mathbf{Y} = - G(\mathbf{Y}^{(m)}),
$$
and updates. Used by `Implicit_nStage_nd_RK_method_generator_semi_analytical` for efficiency.

## How to Use

1. **Define your ODE function** in `functions.py` or elsewhere, ensuring it matches the `ODEFunc` protocol.
2. **Choose a method:**

- For explicit methods like RK4, use `ExplicitRK_method_generator` to create the method function.
- For simple methods like Euler, you can define them directly.
- For implicit methods (any 1-stage or n-stage, arbitrary A and b), use `Implicit_nStage_nd_RK_method_generator_semi_analytical` (preferred, structured Jacobian) or the reference `Implicit_nStage_nd_RK_method_generator_bruteforce`.

3. **Set up the solver:** In `main.py`, specify the initial conditions (`y0`), step size (`h`), and number of steps (`n_steps`).

4. **Run and visualize:** Use `solve_and_plot` from `grapher.py` to run the simulation and plot the results.
