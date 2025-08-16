import matplotlib.pyplot as plt
import numpy as np

from solver import ODESolve


def solve_methods(ode_func, methods, h, n_steps, y0):
    """Run *ODEFunc* with each method and return the trajectories."""
    return [
        ODESolve(h=h, Nsteps=n_steps, y0=y0, f=ode_func, method=m)
        for m in methods
    ]


def print_summary(n_steps, y0, final_t, solutions, labels, t0):
    """Print step info and final values for each method."""
    print(f"Steps {n_steps}, Initial y {y0} Initial t {t0} Final t {final_t}")
    for label, sol in zip(labels, solutions):
        print(f"Final {label}: {sol[-1]}")


def plot_solutions(t, solutions, labels):
    """Plot solutions using matplotlib."""
    fig, ax = plt.subplots()
    for sol, label in zip(solutions, labels):
        ax.plot(t, sol, label=label)
    ax.legend()
    ax.set_title("y(t)")
    plt.show()


def solve_and_plot(labels, ode_func, methods, h, n_steps, y0, t0):
    """Run, print, and plot ODE solutions for all methods."""
    solutions = solve_methods(ode_func, methods, h, n_steps, y0)
    t = np.linspace(t0, h * n_steps, n_steps + 1, endpoint=True)
    print_summary(n_steps, y0, t[-1], solutions, labels, t0)
    plot_solutions(t, solutions, labels)
