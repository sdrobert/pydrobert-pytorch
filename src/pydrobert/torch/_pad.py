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

from typing import Optional, Tuple, overload
from typing_extensions import Literal

import torch

from . import config
from ._compat import script
from ._wrappers import functional_wrapper, proxy


@overload
def pad_variable(
    x: torch.Tensor,
    lens: torch.Tensor,
    pad: torch.Tensor,
    mode: Literal["constant", "reflect", "replicate"] = "constant",
    value: float = config.DEFT_PAD_VALUE,
) -> torch.Tensor:
    ...


@script
def _get_padding_buffers(
    x: torch.Tensor,
    lens: torch.Tensor,
    left_pad: torch.Tensor,
    right_pad: torch.Tensor,
    mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 3
    N, T, F = x.shape
    arange = torch.arange(T, device=x.device)
    if mode == "constant":
        # don't actually do anything. It'll be faster if we just initialize
        # with the fill value
        left_buf = right_buf = x
        # buff = torch.tensor(value, device=device).to(dtype).view(1)
        # left_buf = buff.expand(left_pad.sum() * F)
        # right_buf = buff.expand(right_pad.sum() * F)
    elif mode == "reflect":
        if (left_pad >= lens).any() or (right_pad >= lens).any():
            raise NotImplementedError(
                "For reflect padding, all padding lengths must be less than the "
                "sequence length"
            )
        left_mask = (left_pad.unsqueeze(1) > arange).unsqueeze(2).expand_as(x)
        left_max, right_max = left_pad.max(), right_pad.max()
        left_idxs = (
            (left_pad.unsqueeze(1) - arange[:left_max])
            .clamp_(min=0)
            .unsqueeze(2)
            .expand(N, left_max, F)
        )
        left_buf = x.gather(1, left_idxs).masked_select(left_mask[:, :left_max])
        right_idxs = (
            (lens.unsqueeze(1) - arange[:right_max] - 2)
            .clamp_(min=0)
            .unsqueeze(2)
            .expand(N, right_max, F)
        )
        right_mask = (
            (right_pad.unsqueeze(1) > arange[:right_max])
            .unsqueeze(2)
            .expand(N, right_max, F)
        )
        right_buf = x.gather(1, right_idxs).masked_select(right_mask)
    elif mode == "replicate":
        if (lens < 1).any():
            raise RuntimeError(f"For replicate padding, all lens must be > 0")
        left_mask = (left_pad.unsqueeze(1) > arange).unsqueeze(2).expand_as(x)
        left_max, right_max = left_pad.max(), right_pad.max()
        left_buf = (
            x[:, :1].expand(N, left_max, F).masked_select(left_mask[:, :left_max])
        )
        right_mask_ = (
            (right_pad.unsqueeze(1) > arange[:right_max])
            .unsqueeze(2)
            .expand(N, right_max, F)
        )
        right_buf = (
            x.gather(1, (lens - 1).view(N, 1, 1).expand(N, right_max, F))
            .expand(N, right_max, F)
            .masked_select(right_mask_[:, :right_max])
        )
    else:
        raise ValueError(
            f"mode must be one of 'constant', 'reflect', 'replicate', got '{mode}'"
        )
    return left_buf, right_buf


@script
@functional_wrapper("PadVariable")
def pad_variable(
    x: torch.Tensor,
    lens: torch.Tensor,
    pad: torch.Tensor,
    mode: str = "constant",
    value: float = config.DEFT_PAD_VALUE,
) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError("Expected x to be at least two dimensional")
    shape = x.shape
    N, T = shape[:2]
    if lens.shape != (N,):
        raise ValueError(
            f"For x of shape {shape}, lens should have shape ({N},) but got"
            f"{lens.shape}"
        )
    if pad.shape != (2, N):
        raise ValueError(
            f"For x of shape {shape}, pad should have shape (2, {N}), but got "
            f"{pad.shape}"
        )
    x = x.unsqueeze(-1).flatten(2)
    F = x.size(2)
    left_buf, right_buf = _get_padding_buffers(x, lens, pad[0], pad[1], mode)
    new_lens = lens + pad.sum(0)
    Tp = int(new_lens.max().item())
    arange = torch.arange(max(Tp, T), device=x.device)
    left_mask = (pad[0].unsqueeze(1) > arange[:Tp]).unsqueeze(2).expand(N, Tp, F)
    mid_mask = (
        ((pad[0] + lens).unsqueeze(1) > arange[:Tp]).unsqueeze(2).expand(N, Tp, F)
    )
    right_mask = (new_lens.unsqueeze(1) > arange[:Tp]).unsqueeze(2).expand(N, Tp, F)
    len_mask = (lens.unsqueeze(1) > arange[:T]).unsqueeze(2).expand(N, T, F)
    padded = x.new_full((N, Tp, F), value)
    x = x.masked_select(len_mask)
    padded = padded.masked_scatter(mid_mask & ~left_mask, x)
    if mode != "constant":
        padded = padded.masked_scatter(left_mask, left_buf)
        padded = padded.masked_scatter(right_mask & ~mid_mask, right_buf)
    return padded.view((N, Tp) + shape[2:])


class PadVariable(torch.nn.Module):
    """Pad variable-length input by a variable amount on each side

    This module attempts to replicate the behaviour of :func:`torch.nn.functional.pad`
    on a tensor containing variable sequence lengths with variable amounts of padding.

    Parameters
    ----------
    mode
        How to pad the sequences. :obj:`'constant'`: fill the padding region with the
        value specified by `value`. :obj:`'reflect'`: padded values are reflections
        around the endpoints. For example, the first right-padded value of the ``n``-th
        sequence would be ``x[n, lens[n] - 2``, the third ``x[n, lens[n] - 3]``, and
        so on. :obj:`replicate`: padding duplicates the endpoints of each sequence.
        For example, the left-padded values of the ``n``-th sequence would all be
        ``x[n, 0]``; the right-padded values would be ``x[n, lens[n] - 1]``.
    value
        The value to pad with when ``mode == 'constant'``.
    
    Call Parameters
    ---------------
    x : torch.Tensor
        A tensor of shape ``(N, T, *)`` where ``N`` is the batch index and ``T`` is
        the sequence index.
    lens : torch.Tensor
        A long tensor of shape ``(N,)`` specifying the sequence lengths. Only the values
        in the range ``x[n, :lens[n]]`` are considered part of the sequence of batch
        element ``n``.
    pad : torch.Tensor
        A long tensor of shape ``(2, N)`` specifying how many elements at the start
        (``pad[0]``) and end (``pad[1]``) of each sequence.
    
    Returns
    -------
    padded : torch.Tensor
        A tensor of shape ``(N, T', *)`` such that, for a given batch index ``n``::

            padded[n, :pad[0, n]] = left padding
            padded[n, pad[0,n]:pad[0,n] + lens[n]] = x[n, :lens[n]]
            padded[n, pad[0,n] + lens[n]:pad[0,n] + lens[n] + pad[1, n]] = right padding

    Raises
    ------
    NotImplementedError
        If any value in ``pad[:, n]`` equals or exceeds ``lens[n]`` when
        ``mode == 'reflect'``
    RuntimeError
        If any element in `lens` is less than 1 when ``mode == 'replicate'``

    Examples
    --------

    >>> x = torch.arange(10).view(2, 5)
    >>> x
    tensor([[0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9]])
    >>> lens = torch.tensor([3, 4])
    >>> pad = torch.arange(4).view(2, 2)
    >>> pad.t()  # [[0_left, 0_right], [1_left, 1_right]]
    tensor([[0, 2],
            [1, 3]])
    >>> y = pad_variable(x, lens, pad)  # constant w/ value 0
    >>> y[0, :3 + 0 + 2]
    tensor([0, 1, 2, 0, 0])
    >>> y[1, :4 + 1 + 3]
    tensor([0, 5, 6, 7, 8, 0, 0, 0])
    >>> y = pad_variable(x, lens, pad, 'reflect')
    >>> y[0, :3 + 0 + 2]
    tensor([0, 1, 2, 1, 0])
    >>> y[1, :4 + 1 + 3]
    tensor([6, 5, 6, 7, 8, 7, 6, 5])
    >>> y = pad_variable(x, lens, pad, 'replicate')
    >>> y[0, :3 + 0 + 2]
    tensor([0, 1, 2, 2, 2])
    >>> y[1, :4 + 1 + 3]
    tensor([5, 5, 6, 7, 8, 8, 8, 8])
    """

    __constants__ = ["mode", "value"]

    mode: str
    value: float

    def __init__(
        self,
        mode: Literal["constant", "reflect", "replicate"] = "constant",
        value: float = config.DEFT_PAD_VALUE,
    ):
        super().__init__()
        if mode not in {"constant", "reflect", "replicate"}:
            raise ValueError(
                "mode should be one of 'constant', 'reflect', or 'replicate', got "
                f"'{mode}'"
            )
        self.mode = mode
        self.value = value

    def extra_repr(self) -> str:
        s = f"mode={self.mode}"
        if self.mode == "constant":
            s += f", value={self.value}"
        return s

    def forward(
        self, x: torch.Tensor, lens: torch.Tensor, pad: torch.Tensor
    ) -> torch.Tensor:
        return pad_variable(x, lens, pad, self.mode, self.value)

    __call__ = proxy(forward)


@script
@functional_wrapper("PadMaskedSequence")
def pad_masked_sequence(
    x: torch.Tensor,
    mask: torch.Tensor,
    batch_first: bool = False,
    padding_value: float = config.DEFT_PAD_VALUE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.ndim < 2:
        raise RuntimeError(f"expected x to be at least two-dimensional, got {x.ndim}")
    if mask.ndim != 2:
        raise RuntimeError(f"expected mask to be two-dimensional, got {mask.ndim}")
    if not batch_first:
        x, mask = x.transpose(0, 1), mask.transpose(0, 1)
    lens = mask.sum(1)
    lmask = lens.unsqueeze(1) > torch.arange(x.size(1), device=lens.device)
    lmask = lmask.view(lmask.shape + (1,) * (x.ndim - 2)).expand_as(x)
    mask = mask.view(mask.shape + (1,) * (x.ndim - 2)).expand_as(x)
    x_ = torch.full_like(x, padding_value)
    x_ = x_.masked_scatter(lmask, x.masked_select(mask))
    if not batch_first:
        x_ = x_.transpose(0, 1)
    return x_, lens


class PadMaskedSequence(torch.nn.Module):
    """Select masked elements of tensor, then scatter into right-padded sequences

    Parameters
    ----------
    batch_first
        Whether the first (or second) dimension of `x` is the batch dimension. The
        sequence dimension will be the second (or first).
    padding_value
        The value to right-pad the remaining elements with along the sequence dimension.
    
    Call Parameters
    ---------------
    x : torch.Tensor
        The input tensor. At least two dimensional.
    mask : torch.Tensor
        A boolean tensor whose :obj:`True` values indicate that the associated
        element(s) of `x` should be included in the sequence. Broadcasts with the
        first two dimensions of `x`.
    
    Returns
    -------
    x_ : torch.Tensor
        A tensor of the same shape as `x` such that, supposing ``i`` indexes the
        ``j``-th :obj:`True` element of `mask` for batch index :obj:`n`::

            x_[j, n] = x[i, n]
        
        with the remaining values of `x_` being `padding_value`.
    lens : torch.Tensor
        A vector of the length of the batch dimension which counts the number of
        elements of `x` stored in `x_` per batch element.
    
    Examples
    --------
    >>> x = torch.arange(100).view(10, 10)
    >>> mask = (x % 3) == 0
    >>> pad_masked_sequence = PadMaskedSequence(True, -1)
    >>> x_, lens = pad_masked_sequence(x, mask)
    >>> x_
    tensor([[ 0,  3,  6,  9, -1, -1, -1, -1, -1, -1],
        [12, 15, 18, -1, -1, -1, -1, -1, -1, -1],
        [21, 24, 27, -1, -1, -1, -1, -1, -1, -1],
        [30, 33, 36, 39, -1, -1, -1, -1, -1, -1],
        [42, 45, 48, -1, -1, -1, -1, -1, -1, -1],
        [51, 54, 57, -1, -1, -1, -1, -1, -1, -1],
        [60, 63, 66, 69, -1, -1, -1, -1, -1, -1],
        [72, 75, 78, -1, -1, -1, -1, -1, -1, -1],
        [81, 84, 87, -1, -1, -1, -1, -1, -1, -1],
        [90, 93, 96, 99, -1, -1, -1, -1, -1, -1]])
    >>> lens
    tensor([4, 3, 3, 4, 3, 3, 4, 3, 3, 4])
    >>> x = (x * 2).unsqueeze(2) + torch.arange(2)
    >>> x_, lens = pad_masked_sequence(x, mask)
    >>> x_[:1]
    tensor([[[ 0,  1],
            [ 6,  7],
            [12, 13],
            [18, 19],
            [-1, -1],
            [-1, -1],
            [-1, -1],
            [-1, -1],
            [-1, -1],
            [-1, -1]]])
    """

    __constants__ = ["batch_first", "padding_value"]
    batch_first: bool
    padding_value: float

    def __init__(
        self, batch_first: bool = False, padding_value: float = config.DEFT_PAD_VALUE
    ):
        super().__init__()
        self.batch_first = batch_first
        self.padding_value = padding_value

    def extra_repr(self) -> str:
        return f"batch_first={self.batch_first}, padding_value={self.padding_value}"

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return pad_masked_sequence(x, mask, self.batch_first, self.padding_value)

    __call__ = proxy(forward)


@overload
def chunk_by_slices(
    x: torch.Tensor,
    slices: torch.Tensor,
    lens: Optional[torch.Tensor] = None,
    mode: Literal["constant", "reflect", "replicate"] = "constant",
    value: float = config.DEFT_PAD_VALUE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ...


@script
@functional_wrapper("ChunkBySlices")
def chunk_by_slices(
    x: torch.Tensor,
    slices: torch.Tensor,
    lens: Optional[torch.Tensor] = None,
    mode: str = "constant",
    value: float = config.DEFT_PAD_VALUE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.ndim < 2:
        raise RuntimeError(f"Expected x to be at least 2-dimensional; got {x.ndim}")
    N, T = x.size(0), x.size(1)
    if not N * T:
        return x.new_empty(x.shape), slices.new_zeros((N,))
    rest = x.shape[2:]
    x = x.unsqueeze(-1).flatten(2)
    device = x.device
    if lens is None:
        lens = torch.full((1,), T, dtype=torch.long, device=device).expand(N)
    elif lens.shape != (N,):
        raise RuntimeError(f"Expected lens to be of shape ({N,}); got {lens.shape}")
    F = x.size(2)
    start, end = slices[..., 0].contiguous(), slices[..., 1].contiguous()
    chunk_lens = (end - start).clamp_min_(0)
    empty = chunk_lens == 0
    left_pad = (-start).clamp_min_(0).masked_fill_(empty, 0)
    right_pad = (end - lens).clamp_min_(0).masked_fill_(empty, 0)
    start_ = start.clamp_min(0)
    end_ = torch.min(end, lens)
    slice_lens = (end_ - start_).clamp_min(0)
    left_buf, right_buf = _get_padding_buffers(x, lens, left_pad, right_pad, mode)
    Tp = int(
        torch.max(torch.max(left_pad.max(), chunk_lens.max()), right_pad.max()).item()
    )
    arange = torch.arange(max(T, Tp), device=device)
    slice_mask = (
        ((start.unsqueeze(1) <= arange[:T]) & (end_.unsqueeze(1) > arange[:T]))
        .unsqueeze(-1)
        .expand(N, T, F)
    )
    x = x.masked_select(slice_mask)
    left_mask = (left_pad.unsqueeze(1) > arange[:Tp]).unsqueeze(2).expand(N, Tp, F)
    mid_mask = (
        ((left_pad + slice_lens).unsqueeze(1) > arange[:Tp])
        .unsqueeze(2)
        .expand(N, Tp, F)
    )
    chunks = x.new_full((N, Tp, F), value)
    if mode != "constant":
        chunks = chunks.masked_scatter(left_mask, left_buf)
        right_mask = (
            ((left_pad + slice_lens + right_pad).unsqueeze(1) > arange[:Tp])
            .unsqueeze(2)
            .expand(N, Tp, F)
        )
        right_mask = right_mask & ~mid_mask
        chunks = chunks.masked_scatter(right_mask, right_buf)
        if mode == "reflect":
            # we have to do some extra work for a special case. When the start and
            # end indices are completely contained in the right padding, the slice
            # may start at an offset within the padding. If so, we want to move the
            # start of the slice within the padding to the start of the sequence
            offset = (start_ - lens).clamp_min_(0)
            keep = (offset > 0).view(N, 1, 1)
            right_pad -= offset
            right_mask &= (
                ((left_pad + slice_lens + offset).unsqueeze(1) <= arange[:Tp])
                .unsqueeze(2)
                .expand(N, Tp, F)
            ) & keep
            right_buf = chunks[right_mask]
            right_mask = (
                (right_pad.unsqueeze(1) > arange[:Tp]).unsqueeze(2).expand(N, Tp, F)
            ) & keep
            chunks = chunks.masked_scatter(right_mask & ~mid_mask, right_buf)
    chunks = chunks.masked_scatter(mid_mask & ~left_mask, x)
    return chunks.view((N, Tp) + rest), chunk_lens


class ChunkBySlices(torch.nn.Module):
    """Chunk input using slices, padding where necessary
    
    Parameters
    ----------
    mode
        How to pad slices that go beyond the sequence lengths. See :class:`PadVariable`
        for more information on the modes.
    value
        The value to pad with when ``mode == 'constant'``.
    
    Call Parameters
    ---------------
    x : torch.Tensor
        A tensor of shape ``(N, T, *)`` where ``N`` is the batch index and ``T`` is
        the sequence index.
    slices : torch.Tensor
        A long tensor of shape ``(N, 2)`` containing pairs ``start, end``, where
        `start` and `end` are the start (inclusive) and end (exclusive) indices,
        respectively. Any slices exceeding segment boundaries will be padded according
        to the `mode` specified.
    lens : torch.Tensor, optional
        An optional long tensor of shape ``(N,)`` specifying the sequence lengths. Only
        the values in the range ``x[n, :lens[n]]`` are considered part of the sequence
        of batch element ``n``. If unspecified, all sequences of `x` are assumed to be
        of length ``T``.
    
    Returns
    -------
    chunked : torch.Tensor
        A tensor of shape ``(N, T', *)`` of chunks of `x`. Besides, ``T'``, `chunked`
        matches the shape of `x`.`
    chunked_lens : torch.Tensor
        A long tensor of shape ``(N,)`` with the same interpretation as `lens`, but
        for `chunked` instead.
    
    Warnings
    --------
    Negative indices in slices in Python are usually interpreted as an offset left from
    the end of the sequence. Here, however, negative indices indicate an offset left
    from the start of the sequence. Those values will be interpreted as padding and be
    added to the chunk.

    See Also
    --------
    PadVariable
        For more details on how padding works.
    SliceSpectData
        Can be used to determine `slices` for :class:`SpectDataSet` features. In
        this case, ``x = x[sources]`` and ``lens = lens[sources]`` should be passed
        to this module (using the return value `sources` from :class:`SliceSpectData`).
    ChunkTokenSequenceBySlices
        A similar purpose, but specifically for token sequences from a
        :class:`SpectDataSet`.
    """

    __constants__ = ["mode", "value"]
    mode: str
    value: float

    def __init__(
        self,
        mode: Literal["constant", "reflect", "replicate"] = "constant",
        value: float = config.DEFT_PAD_VALUE,
    ) -> None:
        super().__init__()
        if mode not in {"constant", "reflect", "replicate"}:
            raise ValueError(
                "mode should be one of 'constant', 'reflect', or 'replicate', got "
                f"'{mode}'"
            )
        self.mode = mode
        self.value = value

    def extra_repr(self) -> str:
        s = f"mode={self.mode}"
        if self.mode == "constant":
            s += f", value={self.value}"
        return s

    def forward(
        self,
        x: torch.Tensor,
        slices: torch.Tensor,
        lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return chunk_by_slices(x, slices, lens, self.mode, self.value)

    __call__ = proxy(forward)

