# Copyright 2023 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple, overload
from typing_extensions import Literal

import torch

from . import config
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
    eps: float = config.TINY,
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
        eps: float = config.TINY,
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
    value: float = config.DEFT_PAD_VALUE,
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
        value: float = config.DEFT_PAD_VALUE,
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


@overload
def slice_spect_data(
    in_: torch.Tensor,
    in_lens: Optional[torch.Tensor] = None,
    other_lens: Optional[torch.Tensor] = None,
    policy: Literal["fixed", "ali", "ref"] = None,
    window_type: Literal["symmmetric", "causal", "future"] = "symmetric",
    valid_only: bool = True,
    lobe_size: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ...


@script
@functional_wrapper("SliceSpectData")
def slice_spect_data(
    in_: torch.Tensor,
    in_lens: Optional[torch.Tensor] = None,
    other_lens: Optional[torch.Tensor] = None,
    policy: str = "fixed",
    window_type: str = "symmetric",
    valid_only: bool = True,
    lobe_size: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if in_.ndim < 2:
        raise RuntimeError(f"Expected in_ to be at least 2-dimensional; got {in_.ndim}")
    N, T = in_.shape[:2]
    device = in_.device
    if not T:
        return (
            torch.empty(0, 2, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )
    if lobe_size < 0:
        raise RuntimeError(f"Expected non-negative lobe_size, got {lobe_size}")
    if window_type not in ("symmetric", "causal", "future"):
        raise RuntimeError(
            "expected window_type to be one of 'symmetric', 'casual', or 'future'"
            f"got '{window_type}'"
        )
    if policy == "fixed":
        shift = lobe_size + 1
        if valid_only and window_type == "symmetric":
            window_size = 2 * lobe_size + 1
            starts = torch.arange(0, max(T - window_size + 1, 0), shift, device=device)
            ends = starts + window_size
            mids = ends - 1
        elif window_type == "symmetric":
            window_size = 2 * lobe_size + 1
            half_shift = shift // 2
            TT = (T + half_shift) // shift
            mids = torch.arange(TT, device=device) * shift + half_shift
            starts = mids - window_size // 2
            ends = starts + window_size
        elif valid_only:
            # the behaviour doesn't change with "causal" or "future" when valid_only
            starts = torch.arange(0, max(T - lobe_size, 0), shift, device=device)
            ends = starts + shift
            mids = ends - 1
        elif window_type == "causal":
            starts = torch.arange(-lobe_size, T - lobe_size, shift, device=device)
            ends = starts + shift
            mids = ends - 1
        else:  # future
            starts = mids = torch.arange(0, T, shift, device=device)
            ends = starts + shift
        starts, ends = starts.expand(N, -1), ends.expand(N, -1)
        # starts = starts.clamp_min_(0).expand(N, -1)
        # if in_lens is None:
        #     ends = ends.clamp_max_(T).expand(N, -1)
        # else:
        #     ends = torch.min(ends.unsqueeze(0), in_lens.unsqueeze(1))
        TT = starts.size(1)
        slices = torch.stack([starts, ends], 2).flatten(end_dim=1)
        sources = torch.arange(N, device=device).view(N, 1).expand(N, TT).flatten()
        if in_lens is not None:
            if in_lens.shape != (N,):
                raise RuntimeError(
                    f"Expected in_lens to be of shape ({N},); got {in_lens.shape}"
                )
            mask = (in_lens.unsqueeze(1) > mids).flatten()
            slices = slices[mask]
            sources = sources[mask]
    elif policy == "ali":
        if in_.ndim != 2:
            raise RuntimeError(f"expected tensor of dimension 2 with policy 'ali'")
        mask = in_[:, :-1] != in_[:, 1:]
        arange = torch.arange(T, device=device)
        if in_lens is not None:
            if in_lens.shape != (N,):
                raise RuntimeError(
                    f"Expected in_lens to be of shape ({N},); got {in_lens.shape}"
                )
            mask = mask & (in_lens.view(N, 1) > arange[1:])
        else:
            in_lens = torch.full((N,), T, device=device)
        nonempty = (in_lens > 0).view(N, 1)
        starts = torch.cat([nonempty, mask], 1).nonzero()
        mask = torch.cat([torch.zeros_like(nonempty), mask], 1)
        mask = mask | (nonempty & (in_lens.view(N, 1) == arange))
        ends = mask.nonzero()
        sources = starts[:, 0]
        starts, ends = starts[:, 1], ends[:, 1]
        if lobe_size:
            NN = starts.size(0)
            do_left = window_type in ("symmetric", "causal")
            do_right = window_type in ("symmetric", "future")
            if valid_only:
                offs = (int(do_left) + int(do_right)) * lobe_size
                is_same = sources[: NN - offs] == sources[offs:]
                starts = starts[: NN - offs][is_same]
                ends = ends[offs:][is_same]
                sources = sources[: NN - offs][is_same]
            else:
                start_idx = torch.arange(NN, device=device)
                end_idx = start_idx.clone()
                for n in range(1, lobe_size + 1):
                    offs = (sources[n:] == sources[: NN - n]).long()
                    if do_left:
                        start_idx[n:] -= offs
                    if do_right:
                        end_idx[: NN - n] += offs
                starts = starts[start_idx]
                ends = ends[end_idx]
        slices = torch.stack([starts, ends], 1)
    elif policy == "ref":
        if in_.ndim != 3:
            raise RuntimeError(f"Expected in_ to be 3-dimensional, got {in_.ndim}")
        if in_.size(2) != 3:
            raise RuntimeError(
                f"Expected 3rd dimension of in_ to be of size 3, got {in_.size(2)}"
            )
        starts = in_[..., 1]
        ends = in_[..., 2]
        if in_lens is None:
            in_lens = torch.full((N,), T, device=device)
        if other_lens is None:
            # the final segment's end time
            other_lens = (
                ends[..., 1]
                .gather(1, (in_lens - 1).clamp_min_(0).view(N, 1))
                .squeeze(1)
                .masked_fill_(in_lens == 0, 0)
            )
        elif other_lens.shape != (N,):
            raise RuntimeError(
                f"Expected other_lens to have shape ({N},); got {other_lens.shape}"
            )
        mask = in_lens.view(N, 1) > torch.arange(T, device=device)
        mask &= (in_[..., 1:] >= 0).all(2)
        if window_type in ("symmetric", "causal"):
            starts = starts - lobe_size
        if window_type in ("symmetric", "future"):
            ends = ends + lobe_size
        if valid_only:
            mask &= (starts >= 0) & (ends <= other_lens.view(N, 1))
        else:
            mask &= (ends > 0) & (starts < other_lens.view(N, 1))
        mask &= starts < ends
        starts, ends, mask = starts.flatten(), ends.flatten(), mask.flatten()
        sources = torch.arange(N, device=device).view(N, 1).expand(N, T).flatten()
        starts = starts[mask]
        ends = ends[mask]
        sources = sources[mask]
        slices = torch.stack([starts, ends], 1)
    else:
        raise RuntimeError(
            f"Expected policy to be one of 'fixed', 'ali', or 'ref'; got '{policy}'"
        )
    return slices, sources


class SliceSpectData(torch.nn.Module):
    """Determine slices of feature chunks according to a variety of policies
    
    This module helps to chunk :class:`pydrobert.data.SpectDataLoader` data (or other
    similarly-structured tensors) into smaller units by returning slices of that data.
    The input to this module and the means of determining those slices varies according
    to the `policy` specified (see the notes below for more details). The return values
    can then be used to slice the data. 

    Parameters
    ----------
    policy
        Specifies how to slice the data. If :obj:`'fixed'`, extract windows of fixed
        length at fixed intervals. If :obj:`'ali'`, use changes in frame-level
        alignments to determine segment boundaries and slice along those. If
        :obj:`'ref'`, use token segmentations as slices. See below for more info.
    window_type
        How the window will be constructed around the "middle unit" in the policy. In
        general :obj:`'symmetric'` adds lobes to either side of the middle unit,
        :obj:`'causal'` to the left (towards :obj:`0`), :obj:`'future'` to the right.
    valid_only
        What to do when a would-be slice passes over the length of the data. If
        :obj:`True`, any such slices are thrown out. If :obj:`False`, do something
        dictated by the policy which may preserve the invalid boundaries.
    lobe_size
        Specifies the size of a lobe in the slice's window. When the `policy` is
        :obj:`'fixed'` or :obj:`'ref'`, the unit of `lobe_size` is a single frame. When
        `policy` is :obj:`'ali'`, the unit of `lobe_size` is a whole segment.

    Call Parameters
    ---------------
    in_ : torch.Tensor
        A tensor of shape ``(N, T, *)``, where ``N`` is the batch dimension and ``T`` is
        the (maximum) sequence dimension. When `policy` is :obj:`'fixed'`, `in_` should
        be the batch-first feature tensor `feats` from a
        :class:`pydrobert.data.SpectDataLoader`. When :obj:`'ali'`, `in_` should be the
        batch-first `alis` tensor. When :obj:`'ref'`, `in_` should be the batch-first
        `refs` tensor with segment info.
    in_lens : torch.Tensor, optional
        A long tensor of shape ``(N,)`` specifying the lengths of sequences in `in_`.
        For the ``n``-th batch element, only the elements ``in_[n, :in_lens[n]]`` are
        considered. If unspecified, all sequences are assumed to be of length ``T``.
        For the :obj:`'fixed'` and :obj:`'ali'` policies, this is the `feat_lens`
        tensor from a :class:`pydrobert.data.SpectDataLoader`. When :obj:`'ref'`, it
        is the `ref_lens` tensor.
    other_lens : torch.Tensor, optional
        An additional long tensor of shape ``(N,)`` specifying some other lengths,
        depending on the policy. It is currently only used in the :obj:`'ref'` policy
        and takes the value `feat_lens` from a :class:`pydrobert.data.SpectDataLoader`.

    Returns
    -------
    slices : torch.Tensor
        A long tensor of shape ``(M, 2)`` storing the slices of all batch elements.
        ``M`` is the total number of slices. ``slices[m, 0]`` is the ``m``-th slice's
        start index (inclusive), while ``slices[m, 1]`` is the ``m``-th slice's end
        index (exclusive).
    sources : torch.Tensor
        A long tensor of shape ``(M,)`` where ``sources[m]`` is the batch index of the
        ``m``-th slice.
    
    See Also
    --------
    ChunkBySlices
        Can be used to chunk input using the returned `slices` (after reordering that
        input with `sources`)
    
    Notes
    -----
    If `policy` is :obj:`'fixed'`, slices are extracted at fixed intervals (``lobe_size
    + 1``) along the length of the data. `in_` is assumed to be the data in question,
    e.g. the `feats` tensor in a :class:`pydrobert.data.SpectDataLoader`, in batch-first
    order (although any tensor which matches its first two dimensions will do).
    `in_lens` may be used to specify the actual lengths of the input sequences if they
    were padded to fit in the same batch element. If `window_type` is
    :obj:`'symmetric'`, windows are of size ``1 + 2 * lobe_size``; otherwise, windows
    are of size ``1 + lobe_size``. When `valid_only` is :obj:`True`, slices start at
    index :obj:`0` and as many slices as can be fit fully within the sequences are
    returned. When `valid_only` is :obj:`False` slices are kept if their "middle" index
    lies before the end of the sequence with lobes clamped within the sequence. The
    "middle" index for the symmetric window is at ``slice[0] + window_size // 2``; for
    the causal window it's the last index of the window, ``slice[1] - 1``; for the
    future window it's the first, ``slice[0]``. When `valid_only` is :obj:`False`, the
    initial slice's offsets differ as well: for the symmetric case, it's ``(lobe_size +
    1) // 2 - window_size // 2``; for the causal case, it's :obj:`-lobe_size`; and the
    future case it's still :obj:`0`. As an example, given a sequence of length :obj:`8`,
    the following are the slices under different configurations of the :obj:`'fixed'`
    policy with a `lobe_size` of :obj:`2`::

        [[0, 5], [3, 8]]          # symmetric, valid_only
        [[0, 3], [3, 6]]          # not symmetric, valid_only
        [[-1, 4], [2, 6], [5, 9]] # symmetric, not valid_only
        [[-2, 1], [1, 4], [4, 7]] # causal, not valid_only
        [[0, 3], [3, 6], [6, 9]]  # future, not valid_only
    
    If `policy` is :obj:`'ali'`, slices are extracted from the partition of the sequence
    induced by per-frame alignments. `in_` is assumed to be the alignments in question,
    i.e. the batch-first `alis` tensor in a :class:`pydrobert.data.SpectDataLoader`.
    `in_lens` may be used to specify the actual lengths of the input sequences if they
    were padded to fit in the same batch element. The segments are induced by `ali` as
    follows: a segment starts at index `t` whenever ``t == 0`` or ``alis[n, t - 1] !=
    alis[n, t]``. Slice ``m`` is built from segment ``m`` by starting with the segment
    boundaries and possibly extending the start to the left (towards :obj:`0`) or the
    end to the right (away from :obj:`0`). If `window_type` is :obj:`'symmetric'` or
    :obj:`'causal'`, the ``m``-th segment's start is set to the start of the ``(m -
    lobe_size)``-th. If `window_type` is :obj:`'symmetric'` or :obj:`'future'`, the
    segment's end is set to the end of the ``(m + lobe_size)``-th. Since there are a
    finite number of segments, sometimes either ``(m - lobe_size)`` or ``(m +
    lobe_size)`` will not exist. In that case and if `only_valid` is :obj:`True`, the
    slice is thrown out. If `only_valid` is :obj:`False`, the furthest segment from
    ``m`` in the same direction which also exists will be used. For example, with
    ``in_[n] = [1] * 4 + [2] * 3 + [1] + [5] * 2``, the following are the slices under
    different configurations of the :obj:`'ali'` policy with a `lobe_size` of :obj:`1`::

        [[0, 8], [4, 10]]                   # symmetric, valid_only
        [[0, 7], [4, 8], [7, 10]]           # not symmetric, valid_only
        [[0, 7], [0, 8], [4, 10], [7, 10]]  # symmetric, not valid_only
        [[0, 4], [0, 7], [4, 8], [7, 10]]   # causal, not valid_only
        [[0, 7], [4, 8], [7, 10], [8, 10]]  # future, not valid_only
    
    Finally, if `policy` is :obj:`'ref'`, slices are extracted from a transcription's
    segment boundaries. `in_` is assumed to be the token sequences in question, i.e. the
    batch-first `refs` tensor in a :class:`pydrobert.data.SpectDataLoader`. `in_` should
    be 3-dimensional with the third dimension of size 3: ``in_[..., 0]`` the token
    sequence (ignored), ``in_[..., 1]`` the segment starts (in frames), and ``in_[...,
    2]`` their ends. `in_lens` may be specified to give the length of the token
    sequences (i.e. `ref_lens`). In addition, the lengths of the sequences `in_` is
    segmenting (in frames) may be passed via `other_lens` (i.e. `feat_lens`). The slices
    are built off the available segments. If `window_type` is :obj:`'causal'`,
    `lobe_size` is subtracted from all segments if :obj:`'future'`, `lobe_size` is added
    to all ends; if :obj:`'symmetric'`, both are applied. A segment may be discarded a
    few ways: if either the start or end frame is less than 0 (indicating missing
    segment information); if `in_lens` is set and the token segment is indexed past that
    length (``in_[n, t]`` for any ``t >= in_lens[n]``); the starting frame of a segment
    (after padding) matches or exceeds the ending frame after padding (no empty or
    invalid slices); if :obj:`valid_only` is :obj:`True` and the padded start begins
    before index :obj:`0` or the padded end ends after `other_lens`; and if
    :obj:`valid_only` is :obj:`False` and the padded start begins after `other_lens` or
    ends at or before :obj:`0`. For example, with ``in_[n] = [[1, 0, 0], [2, 2, 3], [3,
    -1, 1], [4, 0, -1], [5, 3, 5], [6, 4, 4]``, `in_lens[n] = 5``, ``other_lens[n] =
    6``, and `lobe_size` of :obj:`2`, the following are the slices under different
    configurations of the :obj:`'ref'` policy::

        [[0, 5]]                  # symmetric, valid_only
        [[0, 3], [1, 5]]          # causal, valid_only
        [[0, 2], [2, 5]]          # future, valid_only
        [[-2, 2], [0, 5], [1, 7]] # symmetric, not valid_only
        [[0, 3], [1, 5]]          # causal, not valid_only
        [[0, 2], [2, 5], [3, 7]]  # future, not valid_only
    """

    __constants__ = ["policy", "window_type", "valid_only", "lobe_size"]

    policy: str
    window_type: str
    valid_only: bool
    lobe_size: int

    def __init__(
        self,
        policy: Literal["fixed", "ali", "ref"] = "fixed",
        window_type: Literal["symmetric", "causal", "future"] = "symmetric",
        valid_only: bool = True,
        lobe_size: int = 0,
    ) -> None:
        super().__init__()
        if policy not in {"fixed", "ali", "ref"}:
            raise ValueError(
                f"policy should be one of 'fixed', 'ali', or 'ref'. Got '{policy}'"
            )
        if window_type not in {"symmetric", "causal", "future"}:
            raise ValueError(
                "window_type should be one of 'symmetric', 'causal', or 'future'. "
                f"Got '{window_type}'"
            )
        if lobe_size < 0:
            raise ValueError(f"lobe_size should be non-negative, got {lobe_size}")
        self.policy, self.window_type, self.lobe_size = policy, window_type, lobe_size
        self.valid_only = valid_only

    def extra_repr(self) -> str:
        return (
            f"policy={self.policy}, window_type={self.window_type}, "
            f"lobe_size={self.lobe_size}, valid_only={self.valid_only}"
        )

    def forward(
        self,
        in_: torch.Tensor,
        in_lens: Optional[torch.Tensor] = None,
        other_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return slice_spect_data(
            in_,
            in_lens,
            other_lens,
            self.policy,
            self.window_type,
            self.valid_only,
            self.lobe_size,
        )

    __call__ = proxy(forward)


@script
@functional_wrapper("ChunkTokenSequencesBySlices")
def chunk_token_sequences_by_slices(
    refs: torch.Tensor,
    slices: torch.Tensor,
    ref_lens: Optional[torch.Tensor] = None,
    partial: bool = False,
    retain: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if refs.ndim == 2:
        return refs.new_empty((0, refs.size(1),)), slices.new_empty((0,))
    elif refs.ndim != 3 or refs.size(2) != 3:
        raise RuntimeError(
            "Expected refs to be 2-dimensional or 3-dimensional with final "
            f"dimension size 3. Got shape '{refs.shape}'"
        )
    N, R = refs.size(0), refs.size(1)
    if slices.shape != (N, 2):
        raise RuntimeError(
            f"Expected slices to be a tensor of shape ({N}, 2), got {slices.shape}"
        )
    arange = torch.arange(R, device=refs.device)
    if ref_lens is None:
        mask = torch.ones((N, R), device=refs.device, dtype=torch.bool)
    elif ref_lens.shape != (N,):
        raise RuntimeError(
            f"Expected ref_lens to be a tensor of shape ({N},), got {ref_lens.shape}"
        )
    else:
        mask = ref_lens.unsqueeze(1) > arange
    mask &= (refs[..., 1:] >= 0).all(2) & (refs[..., 2] >= refs[..., 1])
    if partial:
        # slice_start < ref_end and slice_end > ref_start
        mask &= (slices[..., :1] < refs[..., 2]) & (slices[..., 1:] > refs[..., 1])
    else:
        # slice_start <= ref_start and slice_end >= ref_end
        mask &= (slices[..., :1] <= refs[..., 1]) & (slices[..., 1:] >= refs[..., 2])
    chunked_lens = mask.long().sum(1)
    refs = refs[mask.unsqueeze(2).expand_as(refs)]
    mask = (chunked_lens.unsqueeze(1) > arange).unsqueeze(2).expand(N, R, 3)
    chunked = refs.new_empty((N, R, 3)).masked_scatter_(mask, refs)
    if not retain:
        chunked[..., 1:] += slices[..., 0].view(N, 1, 1).expand(N, R, 2)
    return chunked, chunked_lens


class ChunkTokenSequencesBySlices(torch.nn.Module):
    """Chunk token sequences with segments in slices
    
    Parameters
    ----------
    partial
        If :obj:`True`, a segment of `refs` whose interval partially overlaps with the
        slice will be included in `chunked`. Otherwise, segments in `ref` must fully
        overlap with slices (i.e. be contained within).
    retain
        If :obj:`True`, tokens kept from `refs` will retain their original boundary
        values. Otherwise, boundaries will become relative to the start frame of
        `slices`.
    
    Call Parameters
    ---------------
    refs : torch.Tensor
        A long tensor of shape ``(N, R, 3)`` containing triples ``tok, start, end``,
        where `tok` is the token id, `start` is the start frame (inclusive) of the
        segment, and `end` is its end frame (exclusive). A negative `start` or `end` is
        treated as a missing boundary and will automatically exclude the triple from the
        chunk. `ref` may also be a 2-dimensional long tensor ``(N, R)`` of tokens,
        excluding segment boundaried. However, the return values will always be empty.
    slices : torch.Tensor
        A long tensor of shape ``(N, 2)`` containing pairs ``start, end``, where `start`
        and `end` are the start (inclusive) and end (exclusive) indices, respectively.
    ref_lens : torch.Tensor, optional
        An optional long tensor of shape ``(N,)`` specifying the token sequence lengths.
        Only the values in the range ``refs[n, :ref_lens[n]]`` are considered part of
        the sequence of batch element ``n``. If unspecified, all token sequences of
        `refs` are assumed to be of length ``R``.
    
    Returns
    -------
    chunked : torch.Tensor
        A long tensor of shape ``(N, R', 3)`` of the chunked token sequences.
    chunked_lens : torch.Tensor
        A long tensor of shape ``(N,)`` with the same interpretation as `ref_lens`, but
        for `chunked` instead.
    
    Warnings
    --------
    Negative indices in slices in Python are usually interpreted as an offset left from
    the end of the sequence. In `slices`, however, negative indices indicate an offset
    left from the start of the sequence. In `refs`, negative indices indicate a missing
    boundary and are thrown out. Negative indices in `slices` can impact the returned
    segment boundaries in `chunked`.

    See Also
    --------
    SliceSpectData
        Can be used to determine appropriate `slices`. In this case, ``refs =
        refs[sources]`` and ``ref_lens = ref_lens[sources]`` should be passed to this
        module (using the return value `sources` from :class:`SliceSpectData`).
    ChunkBySlices
        A similar purpose, but for input with an explicit dimension for slicing, such as
        `feats` or `alis` from :class:`SpectDataSet`.
    """

    __constants__ = "partial", "retain"

    partial: bool
    retain: bool

    def __init__(self, partial: bool = False, retain: bool = False) -> None:
        super().__init__()
        self.partial = partial
        self.retain = retain

    def extra_repr(self) -> str:
        if self.partial and self.retain:
            return "partial, retain"
        elif self.partial:
            return "partial"
        elif self.retain:
            return "retain"
        else:
            return ""

    def forward(
        self,
        ref: torch.Tensor,
        slices: torch.Tensor,
        ref_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return chunk_token_sequences_by_slices(
            ref, slices, ref_lens, self.partial, self.retain
        )

    # __call__ = proxy(forward)
