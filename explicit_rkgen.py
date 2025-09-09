import numpy as np
import numpy.typing as npt

from solver import enforce_1d, is_strictly_lower_triangular, ODEFunc


def ExplicitRK_method_generator(A_matrix, b_vec):
    b_vector = enforce_1d(b_vec)
    stage = len(b_vector)
    if not is_strictly_lower_triangular(A_matrix, stage):
        raise ValueError("A has wrong shape")

    def RK_method(y_k, h: np.floating, f: ODEFunc) -> npt.NDArray:
        """_summary_

        Args:
            y_k (npt.NDArray): input y value, should have shape (dim, )
            h (np.floating): float
            f (ODEFunc): takes (dim,) -> (dim,)

        Returns:
            npt.NDArray: output (dim,)
        """

        b = b_vector.copy()
        A = A_matrix.copy()

        dim = len(y_k)
        stages = stage

        assert stage > 0
        assert dim > 0
        assert h > 0

        assert y_k.shape == (dim,)
        assert A.shape == (stages, stages)
        assert b.shape == (stages,)
        assert not isinstance(b, np.matrix)

        # dim is the dimension of our vector ODE, i.e. it's the number of equations
        # stages is the number of stages for our RK method

        Y_Stages = np.zeros((stages, dim))

        K = np.zeros((stages, dim))

        for i in range(stages):
            Y_Stages[i, :] = y_k + h * A[i, :i] @ K[:i, :]

            # note that since A is lower triangular, dotting A[i, :i], K[:i] basically only dots the entries that could be nonzero
            # doing it this way also means that we are basically saying
            # "I know that future stages of Y_i won't be present since this is an explicit method"
            # So hence when I'm computing Y_i I don't need to compute or add f(Y_j), for j>=i
            # So we will only add the values of f(Y_j) for j < i, scaled by the corresponding A coefficients

            # the shape is right I think, since A[i, :i] is 1 x i, K[:i, :] is i x d so I get a 1xd vector out, i.e. exactly the d values for that stage, and I multiply by h and then add y_vec, which is itself a d-dimensional vector

            # intuitively, let's it goes to the i'th row, and since there it's lower triangular, it only takes up to the :i-th value, then does a matrix multiplication with K, specifically the rows of K from 0 to i, which is f(Y) up to the stage that has been calculated so far. It does it with all the dimensions since those should be independent of the stages

            # Then I have K[i, :] = f(Y_Stages[i, :]) which just applies f on the current stage, and does it along the d dimensions, since f should take in a d dimensional vector and spit out a d dimensional vector since this is a d dimensional ODE

            K[i, :] = f(Y_Stages[i, :])

        return y_k + h * b @ K

    return RK_method
