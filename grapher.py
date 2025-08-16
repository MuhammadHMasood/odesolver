import matplotlib.pyplot as plt
from solver import np, ODESolve


def run_methods(the_function, methods, h_val, nsteps, y0):
    """Run ODE solvers for each method and return results."""
    return [
        ODESolve(h=h_val, Nsteps=nsteps, y0=y0, f=the_function, method=m)
        for m in methods
    ]


def print_results(steps, y0, final_t, solutions, labels, t0):
    """Print step info and final values for each method."""
    print(f"Steps {steps}, Initial y {y0} Initial t {t0} Final t {final_t}")
    for label, sol in zip(labels, solutions):
        print(f"Final {label}: {sol[-1]}")


def plot_solutions(exact_t, solutions, labels):
    """Plot solutions using matplotlib."""
    fig, ax = plt.subplots()
    for sol, label in zip(solutions, labels):
        ax.plot(exact_t, sol, label=label)
    ax.legend()
    ax.set_title("y(t)")
    plt.show()


def solve_and_plot(labels, the_function, methods, h_val, nsteps, y0, t0):
    """Run, print, and plot ODE solutions for all methods."""
    solutions = run_methods(the_function, methods, h_val, nsteps, y0)
    exact_t = np.linspace(t0, h_val * nsteps, nsteps + 1, endpoint=True)
    print_results(nsteps, y0, exact_t[-1], solutions, labels, t0)
    plot_solutions(exact_t, solutions, labels)
