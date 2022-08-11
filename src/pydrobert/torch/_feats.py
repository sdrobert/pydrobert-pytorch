# Copyright 2022 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch

from ._compat import script
from ._wrappers import functional_wrapper, proxy


@functional_wrapper("MeanVarianceNormalization")
def mean_var_norm(
    x: torch.Tensor,
    dim: int = -1,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
):
    D = x.ndim
    if dim < -D or dim > D - 1:
        raise IndexError(
            f"Dimension out of range (expected to be in the range of [{-D},{D - 1}], "
            f"got {dim})"
        )
    dim = (dim + D) % D
    X = x.size(dim)
    shape = [1] * D
    shape[dim] = X
    dtype = x.dtype
    if mean is None:
        mean = x.transpose(0, dim).unsqueeze(-1).flatten(1).double().mean(1)
    x = x - mean.view(shape).to(dtype)
    if std is None:
        std = x.transpose(0, dim).unsqueeze(-1).flatten(1).double().std(1, False)
    return (x / std.view(shape).to(x).clamp_min(eps)).to(dtype)


class MeanVarianceNormalization(torch.nn.Module):
    """Normalize features according to mean and variance statistics

    Given input `x`, population mean `mean`, population standard deviation `std`, and
    some small value `eps`, mean-variance normalization for the ``i``-th element of the
    `dim`-th dimension of `x` is defined as

    ::

        y[..., i, ...] = (x[..., i, ...] - mean[i]) / max(std[i], eps).
    
    The `mean` and `std` vectors can be acquired in three ways.
    
    First, they may be passed directly to this module on initialization.
    
    Second, if `mean` and `std` were not specified, they can be estimated from the
    (biased) sample statistics of `x`. This is the same as unit normalization.

    Third, they may be estimated from multiple instances of `x` by accumulating
    sufficient statistics with the :func:`accumulate` method, then writing the biased
    estimates with the :func:`store` method.

    Parameters
    ----------
    dim
        The dimension to be normalized. All other dimensions are considered 
    mean
        If set, a vector representing the population mean. The same size as `std`, if
        specified.
    std
        If set, a vector representing the population standard deviation. The same size
        as `mean`, if specified.
    eps
        A small non-negative floating-point value which ensures nonzero division if
        positive.
    
    Call Parameters
    ---------------
    x : torch.Tensor
        A tensor whose `dim`-th dimension is the same size as `mean` and `std`. To be
        normalized.
    
    Returns
    -------
    y : torch.Tensor
        The normalized tensor of the same shape as `x`.
    
    Examples
    --------
    >>> x = torch.arange(1000, dtype=torch.float).view(10, 10, 10)
    >>> mean = x.flatten(0, 1).double().mean(0)
    >>> std = x.flatten(0, 1).double().std(0, unbiased=False)
    >>> y = MeanVarianceNormalization(-1, mean, std)(x)
    >>> assert torch.allclose(y.flatten(0, 1).mean(0), torch.zeros(1))
    >>> assert torch.allclose(y.flatten(0, 1).std(0, unbiased=False), torch.ones(1))
    >>> mvn = MeanVarianceNormalization()
    >>> y2 = mvn(x)
    >>> assert torch.allclose(y, y2)
    >>> for x_n in x:
    ...     mvn.accumulate(x_n)
    >>> mvn.store()
    >>> assert torch.allclose(mvn.mean, mean)
    >>> assert torch.allclose(mvn.std, std)
    >>> y2 = mvn(x)
    >>> assert torch.allclose(y, y2)
    """

    __constants__ = ["dim", "eps"]
    dim: int
    eps: float
    mean: Optional[torch.Tensor]
    std: Optional[torch.Tensor]

    def __init__(
        self,
        dim: int = -1,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        eps: float = 1e-12,
    ):
        if mean is not None:
            if mean.ndim != 1 or not mean.numel():
                raise ValueError("mean must be a nonempty vector if specified")
        if std is not None:
            if std.ndim != 1 or not std.numel():
                raise ValueError("std must be a nonempty vector if specified")
            if mean is not None and mean.size(0) != std.size(0):
                raise ValueError(
                    "mean and std must be of the same length if both specified, got "
                    f"{mean.size(0)} and {std.size(0)}, respectively"
                )
        if eps < 0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        super().__init__()
        self.dim, self.eps = dim, eps
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("sum", None)
        self.register_buffer("sumsq", None)
        self.register_buffer("count", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mean_var_norm(x, self.dim, self.mean, self.std, self.eps)

    __call__ = proxy(forward)

    @torch.no_grad()
    def accumulate(self, x: torch.Tensor) -> None:
        """Accumulate statistics about mean and variance of input"""
        if self.count is None:
            assert self.sum is None and self.sumsq is None
            X = x.size(self.dim)
            self.count = torch.zeros(1, dtype=torch.double, device=x.device)
            self.sum = torch.zeros(X, dtype=torch.double, device=x.device)
            self.sumsq = torch.zeros(X, dtype=torch.double, device=x.device)
        else:
            assert self.sum is not None and self.sumsq is not None
        x = x.transpose(0, self.dim).unsqueeze(-1).flatten(1)
        self.count += x.size(1)
        self.sum += x.sum(1)
        self.sumsq += x.square().sum(1)

    def store(self, delete_stats: bool = True, bessel: bool = False) -> None:
        """Store mean and variance in internal buffers using accumulated statistics

        Overwrites whatever mean and variance were previously stored in internal buffers
        with those based off calls to :func:`accumulate`.

        Parameters
        ----------
        delete_stats
            Whether to delete the accumulated statistics from internal buffers after the
            mean and variance are stored.
        bessel
            Whether to perform `Bessel's correction
            <https://en.wikipedia.org/wiki/Bessel's_correction>`__ on the variance.
        
        Raises
        ------
        RuntimeError
            If the count of samples is too small to make the estimate. At least one
            accumulated sample is necessary with `bessel` :obj:`False`; two if
            :obj:`True`.
        """
        if self.count is None or (bessel and self.count < 2):
            raise RuntimeError("Too few accumulated statistics")
        assert self.sum is not None and self.sumsq is not None
        self.mean = self.sum / self.count
        var = self.sumsq / self.count - self.mean.square()
        if bessel:
            var *= self.count / (self.count - 1)
        self.std = var.sqrt_()
        if delete_stats:
            self.sum = self.sumsq = self.count = None

