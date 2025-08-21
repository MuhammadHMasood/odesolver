import numpy as np
import numpy.typing as npt

from solver import ODEFunc


# Stage 1 gauss legendre
def implicit_midpoint_1d(y_k, h: np.floating, f: ODEFunc) -> npt.NDArray:
    assert y_k.shape == (1,)

    pass
