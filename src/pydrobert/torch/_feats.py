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
from typing_extensions import Literal

import torch

from ._compat import script, movedim
from ._wrappers import functional_wrapper, proxy


# FIXME(sdrobert): this should be traceable through the module, but this version of
# pytorch (1.8.1) isn't getting it
@script
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
    count: Optional[torch.Tensor]
    sum: Optional[torch.Tensor]
    sumsq: Optional[torch.Tensor]

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

    @torch.jit.export
    def accumulate(self, x: torch.Tensor) -> None:
        """Accumulate statistics about mean and variance of input"""
        if self.count is None:
            assert self.sum is None and self.sumsq is None
            X = x.size(self.dim)
            self.count = torch.zeros(1, dtype=torch.double, device=x.device)
            self.sum = torch.zeros(X, dtype=torch.double, device=x.device)
            self.sumsq = torch.zeros(X, dtype=torch.double, device=x.device)
        # XXX(sdrobert): this is so that torchscript can figure out the type refinement
        count, sum_, sumsq = self.count, self.sum, self.sumsq
        assert (
            isinstance(count, torch.Tensor)
            and isinstance(sum_, torch.Tensor)
            and isinstance(sumsq, torch.Tensor)
        )
        x = x.transpose(0, self.dim).unsqueeze(-1).flatten(1)
        count += x.size(1)
        sum_ += x.sum(1)
        sumsq += x.square().sum(1)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps:e}"

    @torch.jit.export
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
        if self.count is None:
            raise RuntimeError("Too few accumulated statistics")
        count, sum_, sumsq = self.count, self.sum, self.sumsq
        assert (
            isinstance(count, torch.Tensor)
            and isinstance(sum_, torch.Tensor)
            and isinstance(sumsq, torch.Tensor)
        )
        if count < 2:
            raise RuntimeError("Too few accumulated statistics")
        self.mean = mean = sum_ / count
        var = sumsq / count - mean.square()
        if bessel:
            var *= count / (count - 1)
        self.std = var.sqrt_()
        if delete_stats:
            self.sum = self.sumsq = self.count = None


def _feat_delta_filters(order: int, width: int) -> torch.Tensor:
    if order < 0:
        raise RuntimeError(f"order must be non-negative, got {order}")
    if width < 1:
        raise RuntimeError(f"width must be positive, got {width}")
    last_filt = torch.zeros(1 + (2 * width) * order)
    last_filt[width * order] = 1
    filts = [last_filt]
    if order == 0:
        return last_filt.unsqueeze(0)
    kernel = torch.arange(width, -width - 1, -1, dtype=torch.float)
    kernel /= kernel.square().sum()
    for _ in range(order):
        last_filt = torch.nn.functional.conv1d(
            last_filt.view(1, 1, -1), kernel.view(1, 1, -1), padding=width
        ).flatten()
        filts.append(last_filt)
    return torch.stack(filts)


@script
@functional_wrapper("FeatureDeltas")
def feat_deltas(
    x: torch.Tensor,
    dim: int = -1,
    time_dim: int = -2,
    concatenate: bool = True,
    order: int = 2,
    width: int = 2,
    pad_mode: str = "replicate",
    value: float = 0,
    _filters: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if _filters is None:
        _filters = _feat_delta_filters(order, width).to(x)
    else:
        assert _filters.shape == (order + 1, 1 + (2 * width) * order)

    D = x.ndim
    if time_dim < -D or time_dim >= D:
        raise RuntimeError(
            f"Expected dimension 'time_dim' to be in [{-D}, {D-1}], got " f"{time_dim}"
        )
    time_dim = (time_dim + D) % D
    if not concatenate:
        D += 1
    if dim < -D or dim >= D:
        raise RuntimeError(
            f"Expected dimension 'dim' to be in [{-D}, {D-1}], got {dim}"
        )
    dim = (dim + D) % D

    x = x.transpose(time_dim, -1)
    shape = x.shape
    x = x.unsqueeze(0).flatten(0, -2).unsqueeze(1)
    if width:
        x = torch.nn.functional.pad(x, (width * order, width * order), pad_mode, value)
    x = torch.nn.functional.conv1d(x, _filters.unsqueeze(1))
    x = x.view(shape[:-1] + (order + 1,) + shape[-1:])
    x = x.transpose(-2, -1).transpose(time_dim, -2)
    # the order dimension is moving right-to-left, so the original inhabitant of that
    # dimension (if any) should be to its right after the move.
    x = movedim(x, -1, dim)
    if concatenate:
        x = x.flatten(dim, dim + 1)
    return x


class FeatureDeltas(torch.nn.Module):
    r"""Compute deltas of features

    Letting :math:`x` be some input tensor with the `time_dim`-th dimension representing
    the evolution of features over time. Denote that dimension with indices :math:`t`
    and the dimension of the `order` of the deltas with :math:`u`. The :math:`0`-th
    order deltas are just :math:`x` itself; higher order deltas are calculated
    recursively as

    .. math::
    
        x[t, u] = \sum_{w=-width}^{width} x[t + w, u - 1] \frac{w}{\sum_{w'} w'^2}.
    
    Deltas can be seen as a rolling averages: first-order deltas akin to first-order
    derivatives; second-order to second-order, and so on.

    Parameters
    ----------
    dim
        The dimension along which resulting deltas will be stored.
    time_dim
        The dimension along which deltas are calculated.
    concatenate
        If :obj:`True`, delta orders are merged into a single axis with the previous
        occupants of the dimension `dim` via concatenation. Otherwise, a new dimension
        is stacked into the location `dim`.
    order
        The non-negative maximum order of deltas.
    width
        Controls the width of the averaging window.
    pad_mode
        How to pad edges to ensure the same size output. See
        :func:`torch.nn.functional.pad` for more details.
    value
        The value used in constant padding.
    
    Call Parameters
    ---------------
    x : torch.Tensor
    
    Returns
    -------
    deltas : torch.Tensor
        Has the same shape as `x` except for one dimension. If `concatenate` is false,
        a new dimension is inserted into `x` at position `dim` of size ``order + 1``. If
        `concatenate` is true, then the `dim`-th dimension of `deltas` is ``order + 1``
        times the length of that of `x`.
    """

    __constants__ = [
        "dim",
        "time_dim",
        "concatenate",
        "order",
        "width",
        "pad_mode",
        "value",
    ]
    dim: int
    time_dim: int
    order: int
    width: int
    pad_mode: str
    value: float
    filters: torch.Tensor

    def __init__(
        self,
        dim: int = -1,
        time_dim: int = -2,
        concatenate: bool = True,
        order: int = 2,
        width: int = 2,
        pad_mode: Literal["replicate", "constant", "reflect", "circular"] = "replicate",
        value: float = 0.0,
    ):
        if pad_mode not in {"replicate", "constant", "reflect", "circular"}:
            raise ValueError(
                "Expected pad_mode to be one of 'replicate', 'constant', 'reflect', or "
                f"'circular', got '{pad_mode}'"
            )
        super().__init__()
        self.register_buffer("filters", _feat_delta_filters(order, width))
        self.dim, self.time_dim, self.value = dim, time_dim, value
        self.order, self.width, self.pad_mode = order, width, pad_mode
        self.concatenate = concatenate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return feat_deltas(
            x,
            self.dim,
            self.time_dim,
            self.concatenate,
            self.order,
            self.width,
            self.pad_mode,
            self.value,
            self.filters,
        )

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, time_dim={self.time_dim}, "
            f"concatenate={self.concatenate}, order={self.order}, width={self.width}, "
            f"pad_mode={self.pad_mode}, value={self.value}"
        )

    __call__ = proxy(forward)

