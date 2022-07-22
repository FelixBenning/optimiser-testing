# %%
import itertools
from re import I
from xml.dom import IndexSizeErr
import torch

from torch import linalg

# %%
class GaussianRandomField:
    __slots__ = (
        "cov_kernel",
        "remaining_alloc",
        "dim",
        "points",
        "evals",
        "cholesky_cov_L",
    )

    def __init__(self, covariance_kernel):
        self.cov_kernel = covariance_kernel
        self.remaining_alloc = 0
        self.points = []

    def allocate(self, n_points, dim):
        """pre-allocate space for n_points of evaluation, where every evaluation needs space dim"""
        nu_curr = len(self.points)  # currently allocated (unsized) points
        ns_curr = nu_curr * dim  # currently allocated sized space for points
        ns_target = n_points * dim  # allocation size target

        # cholesky
        new_chol = torch.empty(ns_target, ns_target)
        new_chol[:ns_curr, :ns_curr] = self.cholesky_cov_L
        self.cholesky_cov_L = new_chol

        # evaluations
        new_evals = torch.empty(n_points, dim)
        new_evals[:nu_curr] = self.evals
        self.evals = new_evals

        # remaining counter
        self.remaining_alloc = n_points - nu_curr

    def __call__(self, point):
        self.__class__ = EvaluatedGaussianRandomField

        # update points
        self.points.append(point)

        cov = self.cov_kernel(point, point)
        match cov.dim():
            case 0:
                L = torch.sqrt(cov)
                self.dim = 1
                eval = L * torch.randn(1)
            case 2:
                # calculate cholesky for the first time
                L = linalg.cholesky(self.cov_kernel(point, point))
                self.dim = L.size(0)
                eval = L @ torch.randn(self.dim)
            case _:
                raise TypeError(
                    "The covariance kernel does not return a covariance matrix"
                )

        if not self.remaining_alloc:  # no allocation happened
            self.cholesky_cov_L = L
            self.remaining_alloc = 0
            self.evals = torch.stack([eval])
        else:  # pre-allocation happened, save L into preallocation
            self.cholesky_cov_L[: self.dim, : self.dim] = L
            self.remaining_alloc -= 1
            self.evals[0] = eval

        return eval


# %%
class EvaluatedMultinomialGaussianRandomField(GaussianRandomField):
    __slots__ = tuple()

    def __init__(self, covariance_fct, points, evals, dim, cholesky_cov_L=None):
        super().__init__(covariance_fct)
        self.points = points
        self.dim = evals.size(1)

        self.remaining_alloc = 0
        if cholesky_cov_L:
            self.cholesky_cov_L = cholesky_cov_L
        else:
            self.cholesky_cov_L = linalg.cholesky(self.cov_kernel(points))

    def __call__(self, point):
        # allocation and size considerations
        n_points = len(self.points)
        n_sized = n_points * self.dim
        if not self.remaining_alloc:
            self.allocate(2 * n_points, self.dim)
        self.remaining_alloc -= 1

        # get cholesky and covariance
        L = self.cholesky_cov_L[:n_sized, :n_sized]
        C_0 = torch.stack([self.cov_kernel(pt, point) for pt in self.points])

        # update points
        self.points.append(point)

        # update cholesky
        L_ext = linalg.solve_triangular(L, C_0)
        self.cholesky_cov_L[n_sized, :n_sized] = L_ext.T
        linalg.cholesky(
            torch.mm(L_ext, L_ext.T) - self.cov_kernel(point, point),
            out=self.cholesky_cov_L[
                n_sized : n_sized + self.dim, n_sized : n_sized + self.dim
            ],
        )

        # generate evals
        cond_mean = L_ext.T * linalg.solve_triangular(L, self.evals[:n_points])
        new_evals = cond_mean + torch.mm(L_ext, torch.randn(self.dim))

        # update evals
        self.evals[n_points] = new_evals
        return new_evals


# %%
class EvaluatedGaussianRandomField(GaussianRandomField):
    __slots__ = tuple()

    def __init__(self, covariance_fct, points, evals, cholesky_cov_L=None):
        super().__init__(covariance_fct)
        self.points = points
        self.evals = evals
        self.dim = 1

        self.remaining_alloc = 0
        if cholesky_cov_L:
            self.cholesky_cov_L = cholesky_cov_L
        else:
            self.cholesky_cov_L = linalg.cholesky(self.cov_kernel(points))

    def __call__(self, point):
        # allocation and size considerations
        n_points = len(self.points)
        if not self.remaining_alloc:
            self.allocate(2 * n_points, self.dim)
        self.remaining_alloc -= 1

        # get cholesky and covariance
        L = self.cholesky_cov_L[:n_points, :n_points]
        C_0 = torch.stack([self.cov_kernel(pt, point) for pt in self.points])

        # update points
        self.points.append(point)

        # update cholesky
        L_ext = linalg.solve_triangular(
            L,
            C_0.reshape((n_points, 1)),
            upper=False,
            out=self.cholesky_cov_L[n_points, :n_points],
        ).reshape((n_points))
        # self.cholesky_cov_L[n_points, :n_points] = L_ext.T

        cond_var = torch.dot(L_ext, L_ext)
        torch.sqrt(
            self.cov_kernel(point, point) - cond_var,
            out=self.cholesky_cov_L[n_points, n_points],
        )

        # generate evals
        cond_mean = torch.dot(
            L_ext,
            linalg.solve_triangular(L, self.evals[:n_points], upper=False).reshape(
                (n_points)
            ),
        )
        new_evals = cond_mean +  cond_var * torch.randn(self.dim)

        # update evals
        self.evals[n_points] = new_evals
        return new_evals


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


if __name__ == "__main__":
    # %%
    import matplotlib.pyplot as plt

    # %%
    rf = GaussianRandomField(SquaredExponentialKernel())
    rf(torch.zeros(3))
    rf(torch.Tensor([1, 0, 0]))
    rf(torch.Tensor([0, 1, 0]))
    rf(torch.Tensor([0,0,1]))
    print(rf.cholesky_cov_L)
    # %%
    x = torch.Tensor(range(1000))

    plt.plot()
