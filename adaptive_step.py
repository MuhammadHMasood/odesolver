import numpy as np
import numpy.typing as npt
from typing import Protocol, Tuple


from solver import ODEFunc, enforce_1d, is_strictly_lower_triangular


class OneStepAdaptiveNumericalMethod(Protocol):
    def __call__(
        self, y_k, h: np.floating, f: ODEFunc
    ) -> Tuple[npt.NDArray, npt.NDArray]: ...


def AdaptiveODESolve(
    h: np.floating,
    Nsteps: np.integer,
    y0,
    f: ODEFunc,
    method: OneStepAdaptiveNumericalMethod,
    higher_hat_order,
    clamp_lower=0,
    clamp_upper=np.inf,
    h_min=0,
    rtol=1e-3,
    atol=1e-6,
    safety=0.9,
) -> npt.NDArray:
    """
    Solve ODE using a one-step numerical method.
    Returns array of shape (Nsteps+1, dim).
    """
    h_curr = h
    y0 = enforce_1d(y0)
    y_n = np.zeros((Nsteps + 1, len(y0)))
    y_n[0, :] = y0
    i = 0
    time = 0
    while i < Nsteps:
        y_next_hat, y_next = method(y_n[i, :], h_curr, f)
        abserror = np.abs(y_next_hat - y_next)
        scale = atol + rtol * np.maximum(np.abs(y_next_hat), np.abs(y_n[i, :]))
        err = np.sqrt(np.mean((abserror / scale) ** 2))
        if not np.isfinite(err):
            err = 2.0
        if err <= 1:  # accept
            y_n[i + 1, :] = y_next_hat
            i += 1
            time += h
        else:  # reject
            pass

        if err == 0:
            scale_proposed = clamp_upper
        else:
            scale_proposed = safety * (err ** (-1 / (higher_hat_order + 1)))
            scale_proposed = np.clip(scale_proposed, clamp_lower, clamp_upper)

        h_curr = scale_proposed * h_curr
        if h_curr <= h_min:
            raise RuntimeError("Step size underflow")
    return y_n


def AdaptiveODESolve_oneiter(
    h: np.floating,
    y0,
    f: ODEFunc,
    method: OneStepAdaptiveNumericalMethod,
) -> npt.NDArray:
    """
    Solve ODE using a one-step numerical method.
    Returns array of shape (1, dim).
    """
    return method(y0, h, f)


def ExplicitRK_embeddedpair_gen(A_matrix, b_vec, bhat_vec):
    b_vector = enforce_1d(b_vec)
    bhat_vector = enforce_1d(bhat_vec)

    stage = len(b_vector)
    assert len(b_vector) == stage
    if not is_strictly_lower_triangular(A_matrix, stage):
        raise ValueError("A has wrong shape")

    def RK_method(y_k, h: np.floating, f: ODEFunc) -> Tuple[npt.NDArray, npt.NDArray]:
        """_summary_

        Args:
            y_k (_type_): _description_
            h (np.floating): _description_
            f (ODEFunc): _description_

        Returns:
            Tuple[npt.NDArray, npt.NDArray]: _description_
        """

        dim = len(y_k)
        stages = stage

        assert stage > 0
        assert dim > 0
        assert h > 0

        assert y_k.shape == (dim,)
        assert A_matrix.shape == (stages, stages)
        assert b_vector.shape == (stages,)
        # dim is the dimension of our vector ODE, i.e. it's the number of equations
        # stages is the number of stages for our RK method

        Y_Stages = np.zeros((stages, dim))

        K_matrix = np.zeros((stages, dim))

        for i in range(stages):
            Y_Stages[i, :] = y_k + h * A_matrix[i, :i] @ K_matrix[:i, :]
            K_matrix[i, :] = f(Y_Stages[i, :])

        y_next = y_k + h * b_vector @ K_matrix
        y_next_hat = y_k + h * bhat_vector @ K_matrix

        return y_next_hat, y_next

    return RK_method
