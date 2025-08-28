import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import itertools
from functions import f_double_pendulum_v0, f_double_pendulum_v1, f_double_pendulum_v2
from solver import ODESolve, ODESolve_oneiter


def solve_methods(ode_func, methods, h, n_steps, y0):
    """Run *ODEFunc* with each method and return the trajectories."""
    return [ODESolve(h=h, Nsteps=n_steps, y0=y0, f=ode_func, method=m) for m in methods]


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


def animate_double_pendulum_infinite(
    solver_methods, labels, h, y0, mass, gravity, length
):
    fig, axs = plt.subplots(1, 2)

    num_methods = len(solver_methods)

    # y_ns = np.tile([y0], num_methods)

    y_ns = [[y0.copy()] for _ in range(num_methods)]
    energies = [[dp_energy(mass, gravity, length, y0[1:])] for _ in range(num_methods)]

    lines = [[], []]

    for j in range(num_methods):
        lines[0].append(axs[0].plot([], [], label=labels[j])[0])
        lines[1].append(axs[1].plot([], [], label=labels[j])[0])
        pass

    axs[0].set_title("Energy")
    axs[1].set_title("Trajectory (θ vs φ)")
    axs[0].set_xlabel("t")
    axs[0].set_ylabel("Energy")
    axs[1].set_xlabel("θ")
    axs[1].set_ylabel("φ")
    axs[0].legend()
    axs[1].legend()

    def init():
        # i = 0, trajectory
        for group in lines:
            for line in group:
                line.set_data([], [])
        return [line for group in lines for line in group]

    def update(frame):
        for i in range(num_methods):
            y_next = ODESolve_oneiter(
                h, y_ns[i][-1], f_double_pendulum_v1, solver_methods[i]
            )
            y_ns[i].append(y_next)
            energies[i].append(dp_energy(mass, gravity, length, y_next[1:]))

            # arr = np.array(y_ns[i])
            arr = np.stack(
                y_ns[i], axis=0
            )  # safer than .array for multiple method types but it shouldn't matter
            times, thetas, phis = arr[:, 0], arr[:, 1], arr[:, 2]
            # times = arr[:, 0]
            # thetas = arr[:, 1]
            # phis = arr[:, 2]

            energy_vals_i = np.array(energies[i])

            # energy
            lines[0][i].set_data(times, energy_vals_i)

            # trajectories
            lines[1][i].set_data(thetas, phis)

        for ax in axs:
            ax.relim()
            ax.autoscale_view()

        return [line for group in lines for line in group]

    # ani = FuncAnimation(
    #     fig, update, frames=np.arange(200), init_func=init, blit=True, interval=50
    # )

    ani = FuncAnimation(
        fig, update, frames=itertools.count(), init_func=init, blit=False, interval=10
    )
    plt.tight_layout()
    plt.show()

    pass


def dp_energy(mass, gravity, length, y):
    """_summary_

    Args:
        mass (_type_): _description_
        gravity (_type_): _description_
        y (_type_): of the form [theta, phi, theta dot, phi dot]
    """
    theta, phi, thetadot, phidot = y

    kinetic = (
        1
        / 2
        * mass
        * (length**2)
        * (
            (4 / 3) * (thetadot**2)
            + (1 / 3) * (phidot**2)
            + thetadot * phidot * np.cos(theta - phi)
        )
    )
    potential = (
        -mass * gravity * length * ((3 / 2) * np.cos(theta) + (1 / 2) * np.cos(phi))
    )

    return kinetic + potential
