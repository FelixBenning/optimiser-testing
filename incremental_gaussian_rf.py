# %%
import itertools
from re import I
import torch

from torch.linalg import cholesky as cholesky
from torch import linalg

# %%


class GaussianRandomField:
    __slots__ = "cov_fct"

    def __init__(self, covariance_fct):
        self.cov_fct = covariance_fct

    def evaluate(self, points):
        self.__class__ = EvaluatedGaussianRandomField
        self.points = points
        L = cholesky(self.cov_fct(points))
        self.evals = torch.linalg.solve_triangular(L, torch.randn(len(points)))
        self.remaining_alloc = 0
        self.cholesky_cov_L = L
        return self.evals


# %%


class EvaluatedGaussianRandomField(GaussianRandomField):
    __slots__ = "points", "evals", "remaining_alloc", "cholesky_cov_L"

    def __init__(self, covariance_fct, points, evals, cholesky_cov_L=None):
        super().__init__(covariance_fct)
        self.points = points
        self.evals = evals
        self.remaining_alloc = 0
        if cholesky_cov_L:
            self.cholesky_cov_L = cholesky_cov_L
        else:
            self.cholesky_cov_L = cholesky(self.cov_fct(points))

    def evaluate(self, points):
        n = len(self.points)
        L = self.cholesky_cov_L[:n, :n]
        C_0 = self.cov_fct(self.points, points)
        L_ext = torch.linalg.solve_triangular(L, C_0)
        L_ext_transposed = torch.transpose(L_ext)
        cond_covariance = L_ext_transposed * L_ext
        cond_mean = L_ext_transposed * torch.linalg.solve_triangular(L, self.evals)


# %%
class SquaredExponentialKernel(torch.jit.ScriptModule):
    __constants__ = ["variance", "length_scale"]

    def __init__(self, variance=1, length_scale=1):
        super().__init__()
        self.variance = variance
        self.length_scale = length_scale

    def forward(self, x, y):
        diff = x - y
        return self.variance * torch.exp(
            -torch.dot(diff, diff) / (2 * self.length_scale)
        )


class IsotropicCovariance(torch.jit.ScriptModule):
    __constants__ = ["kernel"]

    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel

    def forward(self, points: torch.Tensor):
        n = points.size(-1)
        upper = torch.zeros(n, n)
        for x, y in itertools.combinations(range(n), r=2):
            result[x, y] = self.kernel(x, y)
        result = (
            torch.transpose(upper)
            + upper
            + self.kernel(torch.zeros(1), torch.zeros(1)) * torch.eye(n)
        )


# %%
if __name__ == "__main__":
    # %%
    import matplotlib.pyplot as plt

    # %%
    x = torch.Tensor(range(1000))

    def cov_fct(points, other=None):
        if other:
            pass
        else:

            return torch.exp(-torch.square())

    plt.plot()
