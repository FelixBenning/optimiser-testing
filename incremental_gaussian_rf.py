# %%
from typing import Iterable
import math
import torch
import matplotlib.pyplot as plt

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

    def evaluate(self, points):
        if isinstance(points, Iterable):
            [self(pt) for pt in points]
        else:
            torch.stack([self(pt) for pt in torch.split(points)])

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
        # update points
        self.points.append(point)
        var = self.cov_kernel(point, point)
        match var.dim():
            case 0:
                self.__class__ = EvaluatedGaussianRandomField
                L = torch.sqrt(var)
                self.dim = 1
                eval = L * torch.randn(1)
            case 2:
                self.__class__ = EvaluatedMultinomialGaussianRandomField
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

    def __init__(self, covariance_fct, points, evals, cholesky_cov_L=None):
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
        C_0 = torch.cat([self.cov_kernel(pt, point) for pt in self.points])

        # update points
        self.points.append(point)

        # update cholesky
        L_ext = linalg.solve_triangular(L, C_0, upper=False)
        cond_cov = L_ext @ L_ext.T
        self.cholesky_cov_L[n_sized, :n_sized] = L_ext.T
        linalg.cholesky(
            self.cov_kernel(point, point) - cond_cov,
            out=self.cholesky_cov_L[
                n_sized : n_sized + self.dim, n_sized : n_sized + self.dim
            ],
        )

        # generate evals
        cond_mean = L_ext.T @ linalg.solve_triangular(
            L, self.evals[:n_points].reshape((n_points, 1)), upper=False
        )
        new_evals = cond_mean + linalg.cholesky(cond_cov) @ torch.randn(self.dim)

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
            self.allocate(2 * n_points, 1)
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

        cond_var = torch.sqrt(
            self.cov_kernel(point, point) - torch.dot(L_ext, L_ext),
            out=self.cholesky_cov_L[n_points, n_points],
        )

        # generate evals
        cond_mean = torch.dot(
            L_ext,
            linalg.solve_triangular(L, self.evals[:n_points], upper=False).reshape(
                (n_points)
            ),
        )
        new_evals = cond_mean + torch.sqrt(cond_var) * torch.randn(1)

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
        diff = x - y if x.dim() else (x - y).reshape((1))
        return self.variance * torch.exp(
            -torch.dot(diff, diff) / (2 * (self.length_scale**2))
        )

    def matrix(self, points):
        n = len(points)
        grid = (lambda x: torch.cartesian_prod(x, x))(points).reshape(n, n, 2)
        return torch.tensor([[self(*elm) for elm in row] for row in grid])


# %%
class SquaredExponentialKernelInclDerivative(torch.jit.ScriptModule):
    __constants__ = ["variance", "length_scale"]

    def __init__(self, variance=1, length_scale=1):
        super().__init__()
        self.variance = variance
        self.length_scale = length_scale

    def forward(self, x, y):
        diff = x - y
        dim = 1 if (diff.dim() == 0) else diff.size(0)
        diff = diff.reshape((1, dim))  # row vector
        exponential_factor = self.variance * torch.exp(
            -diff @ diff.T / (2 * (self.length_scale**2))
        )
        c_L = torch.eye(1)  # cov(L, L) up to exponential_factor
        c_grad_L = diff / (self.length_scale**2)
        # cov(grad L,L) up to minus and exponential_factor
        c_grad = -c_grad_L.T @ c_grad_L + torch.eye(dim) / (self.length_scale**2)

        return exponential_factor * torch.cat(
            (
                torch.cat((c_L, c_grad_L), dim=1),
                torch.cat((-c_grad_L.T, c_grad), dim=1),
            )
        )

    def matrix(self, points):
        n = len(points)
        grid = (lambda x: torch.cartesian_prod(x, x))(points).reshape(n, n, 2)
        return torch.cat(
            [torch.cat([self(*elm) for elm in row], dim=1) for row in grid]
        )

# %%
def plotRF(points=200, jitter=0.00001):
    k= SquaredExponentialKernelInclDerivative()
    half = points // 2
    points += int(points % 2 == 0)
    inputs = torch.cat([torch.tensor([0]), torch.linspace(-15, -15/half, half), torch.linspace(15/half, 15, half)])
    L = linalg.cholesky(k.matrix(inputs) + torch.eye(2*points)*jitter)
    iid_normal = torch.randn(2*points)
    pt_and_grads = L @ iid_normal
    pt, grads = pt_and_grads[0::2], pt_and_grads[1::2]

    pt0, grad0 = pt[0], grads[0]
    def first_taylor(x):
        return pt0 + x*grad0
    
    bounds = (torch.min(pt)-pt0)/grad0, (torch.max(pt) - pt0)/grad0
    bounds = min(bounds), max(bounds)

    def mean_after_first_sample(x):
        return (linalg.solve_triangular(L[:2,:2], k(0, x), upper=False).T @ iid_normal[:2])[0]
    
    a = list(zip(inputs, pt))
    a.sort()
    sorted_in, sorted_pt = zip(*a)

    pred = [mean_after_first_sample(x) for x in sorted_in]

    plt.plot(sorted_in, sorted_pt, "b-")
    plt.plot([0], [pt0], "ko", label="Starting Point")
    plt.plot(bounds, [first_taylor(x) for x in bounds], "g--", label="Taylor")
    plt.plot(sorted_in, pred, "r--", label="BLUE")
    plt.legend(loc="upper left")
    plt.show()

plotRF()

# %%
def main():
    pass
    # %%

    # %%
    rf = GaussianRandomField(SquaredExponentialKernel())
    rf(torch.Tensor([1, 0, 0]))
    rf(torch.zeros(3))
    rf(torch.Tensor([0, 1, 0]))
    rf(torch.Tensor([0, 0, 1]))
    rf(torch.Tensor([2, 0, 0]))
    rf(torch.Tensor([0, 2, 0]))
    rf(torch.Tensor([0, 0, 2]))
    print(torch.tril(rf.cholesky_cov_L[:7, :7]))


    # %%
    sqk = SquaredExponentialKernel()
    cov_matrix = sqk.matrix(torch.linspace(0,1,8))
    linalg.cholesky(cov_matrix) # not positive definite ????

    # %%
    x = torch.linspace(0, 10, 30).reshape(30, 1)
    rf = GaussianRandomField(SquaredExponentialKernel())
    y = [rf(pt) for pt in x]
    plt.plot(x, y)

    k = SquaredExponentialKernelInclDerivative()
    cov = k(torch.Tensor([0, 1]), torch.Tensor([0, 0]))

    rf = GaussianRandomField(k)
    rf(torch.Tensor([0]))
    rf(torch.Tensor([0.5]))
    rf(torch.Tensor([1]))


# %%
if __name__ == "__main__":
    main()
