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
                raise TypeError("The covariance kernel does not return a covariance matrix")
        
        if not self.remaining_alloc: # no allocation happened
            self.cholesky_cov_L = L
            self.remaining_alloc = 0
            self.evals = torch.stack([eval])
        else: # pre-allocation happened, save L into preallocation
            self.cholesky_cov_L[:self.dim, :self.dim] = L
            self.remaining_alloc -= 1
            self.evals[0] = eval

        return eval 


# %%
class EvaluatedGaussianRandomField(GaussianRandomField):
    __slots__ = tuple()

    def __init__(self, covariance_fct, points, evals, cholesky_cov_L=None):
        super().__init__(covariance_fct)
        self.points = points
        match evals.dim():
            case 1:  # every evaluation is just a number
                self.dim = 1
            case 2:
                self.dim = evals.size(1)
            case _:
                raise IndexSizeErr("Wrong number of evaluation dimensions")

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

        # update points
        self.points.append(point)

        # get cholesky and covariance
        L = self.cholesky_cov_L[:n_sized, :n_sized]
        C_0 = torch.cat([self.cov_kernel(pt, point) for pt in self.points])

        # update cholesky
        L_ext = linalg.solve_triangular(L, C_0)
        self.cholesky_cov_L[n_sized, :n_sized] = L_ext.T
        linalg.cholesky(
            torch.mm(L_ext, L_ext.T) - self.cov_kernel(point, point),
            out=self.cholesky_cov_L[n_sized, n_sized + self.dim],
        )

        # generate evals
        cond_mean = L_ext.T * linalg.solve_triangular(L, self.evals[:n_points])
        new_evals = cond_mean + torch.mm(L_ext, torch.randn(self.dim))

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


# %%
if __name__ == "__main__":
    # %%
    import matplotlib.pyplot as plt

    # %%
    rf = GaussianRandomField(SquaredExponentialKernel())
    rf(torch.zeros(3))

    # %%
    x = torch.Tensor(range(1000))

    plt.plot()
