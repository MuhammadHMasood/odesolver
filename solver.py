from typing import Protocol

import numpy as np
import numpy.typing as npt


def enforce_1d(x: npt.ArrayLike) -> npt.NDArray:
    """Return *x* as a contiguous 1‑D array."""
    arr = np.asarray(x)
    if arr.ndim == 0:
        return arr[np.newaxis]
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1):
        return arr.ravel()
    raise ValueError(f"Expected scalar or 1D array, got shape {arr.shape}")


def is_strictly_lower_triangular(A: npt.ArrayLike, size: int) -> bool:
    """Return True if *A* is a strictly lower triangular square matrix."""
    mat = np.asarray(A)
    if mat.shape != (size, size):
        return False
    return np.allclose(mat, np.tril(mat, k=-1))


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
