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

from typing import Tuple, overload
from typing_extensions import Literal

import torch

from ._compat import script
from ._wrappers import functional_wrapper, proxy


@overload
def pad_variable(
    x: torch.Tensor,
    lens: torch.Tensor,
    pad: torch.Tensor,
    mode: Literal["constant", "reflect", "replicate"] = "constant",
    value: float = 0.0,
) -> torch.Tensor:
    ...


@script
@functional_wrapper("PadVariable")
def pad_variable(
    x: torch.Tensor,
    lens: torch.Tensor,
    pad: torch.Tensor,
    mode: str = "constant",
    value: float = 0.0,
) -> torch.Tensor:
    old_shape = x.shape
    ndim = len(old_shape)
    if ndim < 2:
        raise ValueError("Expected x to be at least two dimensional")
    N, T = old_shape[:2]
    if lens.shape != (N,):
        raise ValueError(
            f"For x of shape {old_shape}, lens should have shape ({N},) but got"
            f"{lens.shape}"
        )
    if pad.shape != (2, N):
        raise ValueError(
            f"For x of shape {old_shape}, pad should have shape (2, {N}), but got "
            f"{pad.shape}"
        )
    x = x.reshape(N, T, -1)
    F = x.size(2)
    new_lens = lens + pad.sum(0)
    Tp = int(new_lens.max().clamp_(min=T).item())
    arange_ = torch.arange(Tp, device=x.device)
    left_mask = (pad[0].unsqueeze(1) > arange_).unsqueeze(2).expand(N, Tp, F)
    if mode == "constant":
        buff = torch.tensor(value, device=x.device).to(x.dtype).view(1)
        left_pad = buff.expand(pad[0].sum() * F)
        right_pad = buff.expand(pad[1].sum() * F)
    elif mode == "reflect":
        if (pad >= lens.unsqueeze(0)).any():
            raise NotImplementedError(
                "For reflect padding, all padding lengths must be less than the "
                "sequence length"
            )
        max_, _ = pad.max(1)
        left_max, right_max = max_[0], max_[1]
        left_idxs = (
            (pad[0].unsqueeze(1) - arange_[:left_max])
            .clamp_(min=0)
            .unsqueeze(2)
            .expand(N, left_max, F)
        )
        left_pad = x.gather(1, left_idxs).masked_select(left_mask[:, :left_max])
        right_idxs = (
            (lens.unsqueeze(1) - arange_[:right_max] - 2)
            .clamp_(min=0)
            .unsqueeze(2)
            .expand(N, right_max, F)
        )
        right_mask_ = (
            (pad[1].unsqueeze(1) > arange_[:right_max])
            .unsqueeze(2)
            .expand(N, right_max, F)
        )
        right_pad = x.gather(1, right_idxs).masked_select(right_mask_)
    elif mode == "replicate":
        if (lens < 1).any():
            raise RuntimeError(f"For replicate padding, all lens must be > 0")
        max_, _ = pad.max(1)
        left_max, right_max = max_[0], max_[1]
        left_pad = (
            x[:, :1].expand(N, left_max, F).masked_select(left_mask[:, :left_max])
        )
        right_mask_ = (
            (pad[1].unsqueeze(1) > arange_[:right_max])
            .unsqueeze(2)
            .expand(N, right_max, F)
        )
        right_pad = (
            x.gather(1, (lens - 1).view(N, 1, 1).expand(N, right_max, F))
            .expand(N, right_max, F)
            .masked_select(right_mask_[:, :right_max])
        )
    else:
        raise ValueError(
            f"mode must be one of 'constant', 'reflect', 'replicate', got '{mode}'"
        )
    mid_mask = ((pad[0] + lens).unsqueeze(1) > arange_).unsqueeze(2).expand(N, Tp, F)
    len_mask = (lens.unsqueeze(1) > arange_[:T]).unsqueeze(2).expand(N, T, F)
    padded = torch.empty((N, Tp, F), device=x.device, dtype=x.dtype)
    padded = padded.masked_scatter(left_mask, left_pad)
    x = x.masked_select(len_mask)
    padded = padded.masked_scatter(mid_mask & ~left_mask, x)
    right_mask = (new_lens.unsqueeze(1) > arange_).unsqueeze(2).expand(N, Tp, F)
    padded = padded.masked_scatter(right_mask & ~mid_mask, right_pad)
    old_shape = list(old_shape)
    old_shape[1] = Tp
    return padded.view(old_shape)


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
        value: float = 0.0,
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
    padding_value: float = 0.0,
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

    def __init__(self, batch_first: bool = False, padding_value: float = 0.0):
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
