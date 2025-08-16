from solver import ODEFunc, npt, np
from rkgen import ExplicitRK_method_generator
from grapher import solve_and_plot
from functions import f_exponential, f_double_pendulum_v0, f_double_pendulum_v1


# Euler method
def euler_method(y_k, h: np.floating, f: ODEFunc) -> npt.NDArray:
    return y_k + h * f(y_k)


def main():
    # RK4 coefficients
    A_matrix = np.array(
        [[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]], dtype=float
    )
    b_vector = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6], dtype=float)

    RK4 = ExplicitRK_method_generator(A_matrix=A_matrix, b_vector=b_vector)

    Steps = 4
    Final = 1
    h_val = Final / Steps
    y0 = np.array([1])

    print("dy/dy = t, t0 = 0 y(t0)=1, solution y(t) = e^t, t_final = 1")

    solve_and_plot(
        labels=["Euler", "RK4"],
        the_function=f_exponential,
        methods=[euler_method, RK4],
        h_val=h_val,
        nsteps=Steps,
        y0=y0,
        t0=0,
    )


def double_pendulum_sanity_check():
    y0 = np.array([0.0, 0.0, 0.0, 0.1, 0.1])  # t=0, θ=0, φ=0, θ̇=0.1, φ̇=0.1
    print("Double Pendulum Sanity Check, v0, v1 (should be same)")
    print("y = [t, θ, φ, θ̇, φ̇]")
    print(f"v0: f({y0}) = {f_double_pendulum_v0(y0)}")
    print(f"v1: f({y0}) = {f_double_pendulum_v0(y0)}")


if __name__ == "__main__":
    # main()
    double_pendulum_sanity_check()
