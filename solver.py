from typing import Protocol
import numpy as np
import numpy.typing as npt


def enforce_1d(x):
    """Ensure input is a 1D numpy array."""
    x = np.asarray(x)
    if x.ndim == 0:
        return x[np.newaxis]
    elif x.ndim == 1:
        return x
    elif x.ndim == 2:
        if x.shape[0] == 1 or x.shape[1] == 1:
            return x.ravel()
    raise ValueError(f"Expected scalar or 1D array, got shape {x.shape}")


def is_strictly_lower_triangular_and_sbys(A, s):
    """Check if A is strictly lower triangular and square of size s."""
    A = np.asarray(A)
    if A.shape != (s, s):
        return False
    return np.allclose(A, np.tril(A, k=-1))


class ODEFunc(Protocol):
    def __call__(self, y) -> npt.NDArray: ...


class OneStepNumericalMethod(Protocol):
    def __call__(self, y_k, h: np.floating, f: ODEFunc) -> npt.NDArray: ...


def ODESolve(
    h: np.floating,
    Nsteps: np.integer,
    y0,
    f: ODEFunc,
    method: OneStepNumericalMethod,
) -> npt.NDArray:
    """
    Solve ODE using a one-step numerical method.
    Returns array of shape (Nsteps+1, dim).
    """
    y0 = enforce_1d(y0)
    y_n = np.zeros((Nsteps + 1, len(y0)))
    y_n[0, :] = y0
    for i in range(Nsteps):
        y_n[i + 1, :] = method(y_n[i, :], h, f)
    return y_n
